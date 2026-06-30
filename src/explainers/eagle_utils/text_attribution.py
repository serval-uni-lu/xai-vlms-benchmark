import numpy as np
from typing import Any
import torch
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class UniversalEAGLETextAdaptor(nn.Module):
    """
    Bridges EAGLE's Text Explainer with BaseVLMWrapper.
    """
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper
        self.device = wrapper.device
        
        self.tokenizer = self.wrapper.processor.tokenizer
        
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        if self.tokenizer.pad_token_id is not None:
            self.mask_token_id = self.tokenizer.pad_token_id
        else:
            self.mask_token_id = self.tokenizer.eos_token_id
            
        if self.mask_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id for masking.")
            
        # 1. Initialize as an integer (0) instead of None
        self.prompt_length: int = 0
        
        # 2. Empty 1D tensor for generated text IDs
        self.generated_answer_ids: torch.Tensor = torch.empty(0, dtype=torch.long)
        
        # 3. Empty 1D tensor for absolute target positions
        self.target_token_position: torch.Tensor = torch.empty(0, dtype=torch.long)
        
        # 4. Empty 1D tensor for the target vocabulary integer IDs
        self.selected_interpretation_token_word_id: torch.Tensor = torch.empty(0, dtype=torch.long)
        
        # 5. Type-hinted dictionary for the cached visual features
        self.static_vision_inputs: dict[str, Any] = {}

    def forward(self, input_ids, attention_mask=None):
        single_input = input_ids.dim() == 1
        if single_input:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(self.device, dtype=torch.long)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(self.device, dtype=torch.long)

        # Append the generated prefix up to the target token
        max_target_position = int(self.target_token_position.max().item())
        generated_prefix_len = max(0, max_target_position - self.prompt_length)
        
        if generated_prefix_len > 0:
            generated_prefix = self.generated_answer_ids[:generated_prefix_len]
            prefix_batch = generated_prefix.unsqueeze(0).expand(input_ids.shape[0], -1)
            
            input_ids = torch.cat([input_ids, prefix_batch], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(prefix_batch, device=self.device)],
                dim=1
            )

        # ---------------------------------------------------------
        # FIX 1: Qwen-Specific Sequence Alignment
        # ---------------------------------------------------------
        if "mm_token_type_ids" in self.static_vision_inputs:
            mm_token_type_ids = self.static_vision_inputs["mm_token_type_ids"]
            if mm_token_type_ids.shape[1] < input_ids.shape[1]:
                text_tail = torch.zeros(
                    (mm_token_type_ids.shape[0], input_ids.shape[1] - mm_token_type_ids.shape[1]),
                    dtype=mm_token_type_ids.dtype,
                    device=self.device
                )
                self.static_vision_inputs["mm_token_type_ids"] = torch.cat([mm_token_type_ids, text_tail], dim=1)

        for stale_key in ("token_type_ids", "position_ids", "cache_position"):
            self.static_vision_inputs.pop(stale_key, None)

        # ---------------------------------------------------------
        # FIX 2: Micro-Batching for VLM Feature Alignment
        # ---------------------------------------------------------
        all_selected_probabilities = []
        
        # Iterate through EAGLE's perturbed batch one by one
        for i in range(input_ids.shape[0]):
            single_input_ids = input_ids[i:i+1] # Slice to keep 2D shape (1, Seq_Len)
            single_attention_mask = attention_mask[i:i+1]
            
            forward_kwargs = {
                "input_ids": single_input_ids,
                "attention_mask": single_attention_mask,
                "return_full_sequence": True,
                **self.static_vision_inputs 
            }

            with torch.no_grad():
                logits = self.wrapper(**forward_kwargs)

            # Extract target probabilities
            target_logits = logits[:, self.target_token_position - 1, :]
            probabilities = torch.nn.functional.softmax(target_logits, dim=-1)

            selected_token_ids = self.selected_interpretation_token_word_id.to(self.device)
            gather_indices = selected_token_ids.unsqueeze(0).unsqueeze(-1)
            
            selected_probs = probabilities.gather(dim=2, index=gather_indices).squeeze(-1)
            all_selected_probabilities.append(selected_probs)
            
        # Stack all results back together to satisfy EAGLE's return expectation
        batched_probabilities = torch.cat(all_selected_probabilities, dim=0)
        
        return batched_probabilities[0] if single_input else batched_probabilities


class EfficientLLMSubModularExplanationText(object):
    """Black-box submodular attribution over text token-span masks."""

    def __init__(
        self,
        model: UniversalEAGLETextAdaptor,
        lambda1=1.0,
        lambda2=1.0,
        search_scope=10,
        pending_samples=8,
        update_step=5,
        batch_size=None,
        mask_strategy="replace",
        mask_token_id=None,
    ):
        self.LLM = model
        self.device = self.LLM.device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.search_scope = search_scope
        self.pending_samples = pending_samples
        self.update_step = update_step
        self.batch_size = batch_size
        self.mask_strategy = mask_strategy
        self.mask_token_id = mask_token_id if mask_token_id is not None else self.LLM.mask_token_id

    def save_file_init(self):
        self.saved_json_file = {
            "insertion_score": [],
            "deletion_score": [],
            "smdl_score": [],
            "insertion_word_score": [],
            "deletion_word_score": [],
            "region_area": [],
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "mask_strategy": self.mask_strategy,
            "insertion_score_oppose": [],
            "deletion_score_oppose": [],
            "insertion_word_score_oppose": [],
            "deletion_word_score_oppose": [],
            "region_area_oppose": [],
            "selected_region_indices": [],
            "selected_region_indices_oppose": [],
        }

    def _to_float_masks(self, masks):
        if isinstance(masks, torch.Tensor):
            return masks.to(self.device, dtype=torch.float32)
        return torch.from_numpy(np.asarray(masks)).to(self.device, dtype=torch.float32)

    def _visible_to_inputs(self, visible_masks):
        visible_masks = visible_masks.clamp(0, 1)
        source_ids = self.source_input_ids.unsqueeze(0).expand(visible_masks.shape[0], -1)

        if self.mask_strategy == "attention":
            input_ids = source_ids
            attention_mask = visible_masks.to(torch.long)
        else:
            mask_ids = torch.full_like(source_ids, int(self.mask_token_id))
            input_ids = torch.where(visible_masks.bool(), source_ids, mask_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    def LLM_inference_batch(self, visible_masks):
        if visible_masks.shape[0] == 0:
            return torch.empty(0, device=self.device)

        batch_size = self.batch_size or visible_masks.shape[0]
        results = []
        for start in range(0, visible_masks.shape[0], batch_size):
            end = min(start + batch_size, visible_masks.shape[0])
            input_ids, attention_mask = self._visible_to_inputs(visible_masks[start:end])
            output_logits = self.LLM(input_ids, attention_mask=attention_mask)
            if output_logits.dim() == 1:
                output_logits = output_logits.unsqueeze(0)
            results.append(output_logits)
        return torch.cat(results, dim=0)

    def _paired_inference(self, insertion_visible, deletion_visible):
        paired_masks = torch.cat([insertion_visible, deletion_visible], dim=0)
        paired_scores = self.LLM_inference_batch(paired_masks).to(torch.float32)
        split_at = insertion_visible.shape[0]
        return paired_scores[:split_at], paired_scores[split_at:]

    def save_positive_file(self, insertion_score, deletion_score, smdl_score, region_index):
        self.saved_json_file["insertion_score"].append(insertion_score.mean().cpu().numpy().item())
        self.saved_json_file["insertion_word_score"].append(insertion_score.cpu().numpy().tolist())
        self.saved_json_file["deletion_score"].append(deletion_score.mean().cpu().numpy().item())
        self.saved_json_file["deletion_word_score"].append(deletion_score.cpu().numpy().tolist())
        self.saved_json_file["smdl_score"].append(smdl_score.cpu().item())
        self.saved_json_file["selected_region_indices"].append(int(region_index))

    def save_negative_file(self, insertion_score, deletion_score, region_index):
        self.saved_json_file["insertion_score_oppose"].append(insertion_score.mean().cpu().numpy().item())
        self.saved_json_file["insertion_word_score_oppose"].append(insertion_score.cpu().numpy().tolist())
        self.saved_json_file["deletion_score_oppose"].append(deletion_score.mean().cpu().numpy().item())
        self.saved_json_file["deletion_word_score_oppose"].append(deletion_score.cpu().numpy().tolist())
        self.saved_json_file["selected_region_indices_oppose"].append(int(region_index))

    def _candidate_tensors(self, candidates):
        masks = self._to_float_masks([candidate["mask"] for candidate in candidates])
        indices = [candidate["index"] for candidate in candidates]
        return masks, indices

    def evaluation_maximum_sample(self, S_set, S_set_opposite):
        candidates = self.V_set
        if len(S_set) and self.update_count % self.update_step != 0:
            candidates = candidates[: self.search_scope]

        candidate_masks, candidate_indices = self._candidate_tensors(candidates)
        selected_candidate_masks = (candidate_masks + self.refer_baseline.unsqueeze(0)).clamp(0, 1)
        insertion_visible = (self.fixed_visible_mask.unsqueeze(0) + selected_candidate_masks).clamp(0, 1)
        deletion_visible = (1 - selected_candidate_masks).clamp(0, 1)

        with torch.no_grad():
            insertion_scores, deletion_scores = self._paired_inference(
                insertion_visible, deletion_visible
            )
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
            smdl_scores = smdl_scores.mean(-1)
            arg_max_index = int(smdl_scores.argmax().cpu().item())

            if len(S_set) == 0 or self.update_count % self.update_step == 0:
                sorted_indices = torch.argsort(smdl_scores, descending=True).cpu().tolist()
                self.V_set = [candidates[i] for i in sorted_indices] + self.V_set[len(candidates) :]
                chosen_local_index = 0
                score_index = sorted_indices[0]
            else:
                chosen_local_index = arg_max_index
                score_index = arg_max_index

            chosen = self.V_set[chosen_local_index]
            chosen_mask = self._to_float_masks(chosen["mask"])
            S_set.append(chosen["mask"])
            self.refer_baseline = (self.refer_baseline + chosen_mask).clamp(0, 1)
            del self.V_set[chosen_local_index]

            self.save_positive_file(
                insertion_scores[score_index],
                deletion_scores[score_index],
                smdl_scores[score_index],
                chosen["index"],
            )
            self.saved_json_file["region_area"].append(
                (self.refer_baseline.sum() / self.region_area).cpu().item()
            )

            if len(S_set_opposite) == 0 and self.V_set:
                remaining_candidate_ids = {id(item) for item in self.V_set}
                remaining_score_indices = [
                    index for index, item in enumerate(candidates) if id(item) in remaining_candidate_ids
                ]
                if remaining_score_indices:
                    remaining_scores = smdl_scores[remaining_score_indices]
                    arg_min_position = int(remaining_scores.argmin().cpu().item())
                    arg_min_index = remaining_score_indices[arg_min_position]
                    weakest = candidates[arg_min_index]
                else:
                    arg_min_index = score_index
                    weakest = self.V_set[-1]

                weakest_list_index = next(
                    index for index, item in enumerate(self.V_set) if id(item) == id(weakest)
                )
                del self.V_set[weakest_list_index]
                weakest_mask = self._to_float_masks(weakest["mask"])
                S_set_opposite.append(weakest["mask"])
                self.refer_baseline_opposite = (self.refer_baseline_opposite - weakest_mask).clamp(0, 1)
                self.save_negative_file(
                    deletion_scores[arg_min_index],
                    insertion_scores[arg_min_index],
                    weakest["index"],
                )
                self.saved_json_file["region_area_oppose"].append(
                    (
                        (self.fixed_visible_mask.sum() + self.refer_baseline_opposite.sum())
                        / self.total_visible_area
                    )
                    .cpu()
                    .item()
                )
                return S_set, S_set_opposite

            if len(self.V_set) > self.pending_samples:
                oppose_candidates = self.V_set[-self.pending_samples :]
                oppose_masks, oppose_indices = self._candidate_tensors(oppose_candidates)
                remaining_masks = (
                    self.refer_baseline_opposite.unsqueeze(0) - oppose_masks
                ).clamp(0, 1)
                insertion_visible = (
                    self.fixed_visible_mask.unsqueeze(0) + remaining_masks
                ).clamp(0, 1)
                deletion_visible = (1 - remaining_masks).clamp(0, 1)
                insertion_scores, deletion_scores = self._paired_inference(
                    insertion_visible, deletion_visible
                )
                oppose_smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (
                    1 - deletion_scores
                )
                arg_max_index_oppose = int(oppose_smdl_scores.mean(-1).argmax().cpu().item())
                chosen_oppose = oppose_candidates[arg_max_index_oppose]
                chosen_oppose_mask = self._to_float_masks(chosen_oppose["mask"])
                S_set_opposite.append(chosen_oppose["mask"])
                self.refer_baseline_opposite = (
                    self.refer_baseline_opposite - chosen_oppose_mask
                ).clamp(0, 1)
                del self.V_set[-self.pending_samples + arg_max_index_oppose]
                self.save_negative_file(
                    insertion_scores[arg_max_index_oppose],
                    deletion_scores[arg_max_index_oppose],
                    oppose_indices[arg_max_index_oppose],
                )
                self.saved_json_file["region_area_oppose"].append(
                    (
                        (self.fixed_visible_mask.sum() + self.refer_baseline_opposite.sum())
                        / self.total_visible_area
                    )
                    .cpu()
                    .item()
                )

        return S_set, S_set_opposite

    def get_merge_set(self):
        S_set = []
        S_set_opposite = []
        self.refer_baseline = torch.zeros_like(self.explainable_mask, device=self.device)
        self.refer_baseline_opposite = self.explainable_mask.clone()

        baseline_visible = self.fixed_visible_mask.unsqueeze(0)
        original_visible = torch.ones_like(self.fixed_visible_mask).unsqueeze(0)
        scores = self.LLM_inference_batch(torch.cat([baseline_visible, original_visible], dim=0)).to(
            torch.float32
        )
        self.saved_json_file["baseline_score"] = scores[0].cpu().numpy().tolist()
        self.saved_json_file["org_score"] = scores[1].cpu().numpy().tolist()

        self.update_count = 0
        for _ in tqdm(range(self.saved_json_file["sub-region_number"])):
            if len(self.V_set) == 1:
                last = self.V_set.pop(0)
                S_set.append(last["mask"])
                self.saved_json_file["selected_region_indices"].append(int(last["index"]))
                break
            if len(self.V_set) == 0:
                break
            S_set, S_set_opposite = self.evaluation_maximum_sample(S_set, S_set_opposite)
            self.update_count += 1

        self.saved_json_file["insertion_score"] = (
            self.saved_json_file["insertion_score"]
            + self.saved_json_file["insertion_score_oppose"][::-1]
            + [scores[1].cpu().mean().item()]
        )
        self.saved_json_file["deletion_score"] = (
            self.saved_json_file["deletion_score"]
            + self.saved_json_file["deletion_score_oppose"][::-1]
            + [scores[0].cpu().mean().item()]
        )
        self.saved_json_file["insertion_word_score"] = (
            self.saved_json_file["insertion_word_score"]
            + self.saved_json_file["insertion_word_score_oppose"][::-1]
            + [scores[1].cpu().numpy().tolist()]
        )
        self.saved_json_file["deletion_word_score"] = (
            self.saved_json_file["deletion_word_score"]
            + self.saved_json_file["deletion_word_score_oppose"][::-1]
            + [scores[0].cpu().numpy().tolist()]
        )
        self.saved_json_file["region_area"] = (
            self.saved_json_file["region_area"]
            + self.saved_json_file["region_area_oppose"][::-1]
            + [1.0]
        )
        self.saved_json_file["selected_region_indices"] = (
            self.saved_json_file["selected_region_indices"]
            + self.saved_json_file["selected_region_indices_oppose"][::-1]
        )

        ordered_sets = S_set + S_set_opposite[::-1]
        merged_set = np.asarray(ordered_sets, dtype=np.float32)
        self.saved_json_file["smdl_score"] = (
            np.array(self.saved_json_file["insertion_score"])
            + 1
            - np.array(self.saved_json_file["deletion_score"])
        ).tolist()
        return merged_set

    def __call__(self, source_input_ids, V_set):
        self.save_file_init()
        self.saved_json_file["sub-region_number"] = len(V_set)

        self.source_input_ids = torch.as_tensor(source_input_ids, device=self.device, dtype=torch.long)
        self.V_set = [
            {"index": index, "mask": np.asarray(mask, dtype=np.float32)}
            for index, mask in enumerate(V_set)
        ]
        if len(self.V_set) == 0:
            raise ValueError("V_set is empty; no text spans were found for attribution.")

        V_set_tensor = self._to_float_masks([item["mask"] for item in self.V_set])
        self.explainable_mask = V_set_tensor.max(dim=0).values.clamp(0, 1)
        self.fixed_visible_mask = (1 - self.explainable_mask).clamp(0, 1)
        self.region_area = self.explainable_mask.sum().clamp_min(1.0)
        self.total_visible_area = torch.tensor(
            float(self.source_input_ids.numel()), device=self.device, dtype=torch.float32
        ).clamp_min(1.0)

        return self.get_merge_set(), self.saved_json_file

