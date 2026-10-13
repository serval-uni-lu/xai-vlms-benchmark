import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import ImageFilter
from torch import Tensor

from src.metrics.base import BaseMetric
from src.models import BaseVLMWrapper
from src.utils.faithfulness_utils import (
    _reshape_pixels_back_faithfulness,
    _reshape_pixels_faithfulness,
    get_most_important_tokens_multimodal,
    get_most_important_tokens_pixel,
    get_most_important_tokens_token,
    get_text_mask,
    make_blur_baseline,
    score_output,
)


# Helper to safely convert arrays/tensors of shape (S, B) to flat Python lists
def safe_flatten_to_list(array_like):
    if hasattr(array_like, "flatten"):
        return array_like.flatten().tolist()
    elif hasattr(array_like, "tolist"):
        flat_list = []
        for item in array_like.tolist():
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        return flat_list
    return list(array_like)


class FaithfulnessMetric(BaseMetric):
    def __init__(
        self,
        perturbation_steps: list[float],
        pad_token_id: int,
        special_token_ids: list[int],
        semantic_mask: Tensor | None = None,
        mask_value: float = 0.0,
        filter_keywords: bool = True,
    ):
        super().__init__("Faithfulness")
        self.steps = perturbation_steps
        self.pad_token_id = pad_token_id
        self.special_token_ids = special_token_ids or []
        self.mask_value = mask_value
        self.filter_keywords = filter_keywords
        self.semantic_mask = semantic_mask

    def set_semantic_mask(self, new_mask):
        self.semantic_mask = new_mask

    def _evaluate_over_tokens(self, eval_func, num_tokens: int, **kwargs) -> dict[str, Any]:
        """
        Runs the perturbation function for EACH targeted token's attribution map individually,
        then averages the resulting AUCs and Curves.
        """
        aggregated_results = {}

        for i in range(num_tokens):
            step_kwargs = kwargs.copy()

            # Slice the attributions to shape [1, ...] to satisfy the eval function
            if "pixel_attribution" in step_kwargs and step_kwargs["pixel_attribution"] is not None:
                step_kwargs["pixel_attribution"] = step_kwargs["pixel_attribution"][i : i + 1]

            if "token_attribution" in step_kwargs and step_kwargs["token_attribution"] is not None:
                step_kwargs["token_attribution"] = step_kwargs["token_attribution"][i : i + 1]

            # Run the underlying masking function for just this one token
            res = eval_func(**step_kwargs)

            # Collect results
            for k, v in res.items():
                if k not in aggregated_results:
                    aggregated_results[k] = []
                aggregated_results[k].append(v)

        # Average the results across all tokens
        final_results = {}
        for k, v_list in aggregated_results.items():
            # Convert tensors to numpy arrays safely
            v_list_np = [v.detach().cpu().numpy() if hasattr(v, "detach") else np.array(v) for v in v_list]
            # Stack and compute the mean (works perfectly for both scalar AUCs and array Curves)
            final_results[k] = np.mean(np.stack(v_list_np), axis=0)

        return final_results

    def compute(
        self,
        wrapper,  # BaseVLMWrapper
        sample: dict[str, Any],
        xai_result: dict[str, Any],
        required_metrics=None,
        compute_syn_only=False,
    ) -> dict[str, Any]:

        inputs = xai_result["inputs"]
        target_ids = xai_result["target_ids"]
        pixel_attr = xai_result.get("pixel_attribution")
        tok_attr = xai_result.get("token_attribution")

        # Determine how many tokens we are evaluating
        num_tokens = 1
        if pixel_attr is not None:
            num_tokens = pixel_attr.shape[0]
        elif tok_attr is not None:
            num_tokens = tok_attr.shape[0]

        image = sample.get("image")
        # question = str(sample.get("question"))
        blur_baseline = None
        if image is not None:
            text = sample.get("text")
            blurred_img = image.filter(ImageFilter.GaussianBlur(radius=30))
            blurred_inpts = wrapper.get_inputs(blurred_img, text)
            blur_baseline = blurred_inpts["pixel_values"]
        else:
            blur_baseline = torch.full_like(inputs["pixel_values"],
                                            self.mask_value).to(wrapper.device)

        results = {}

        # Get text only mask
        model_type = getattr(wrapper.model.config, "model_type", "").lower()
        semantic_mask = get_text_mask(inputs["input_ids"], model_type, wrapper.processor.tokenizer)

        # --- A. Image Perturbation ---
        if pixel_attr is not None and not compute_syn_only:
            start_time = time.perf_counter()

            img_res = self._evaluate_over_tokens(
                eval_image_perturbation_batch,  # Ensure this is imported in your file!
                num_tokens=num_tokens,
                model=wrapper,
                inputs=inputs,
                target_ids=target_ids,
                pixel_attribution=pixel_attr,
                perturbation_steps=self.steps,
                blur_baseline=blur_baseline,
                mask_value=self.mask_value,
                filter_keywords=self.filter_keywords,
            )

            results["time_img_pert"] = time.perf_counter() - start_time
            results.update(self._format_results("img", img_res, required_metrics=required_metrics))

        # --- B. Token Perturbation ---
        if tok_attr is not None and not compute_syn_only:
            start_time = time.perf_counter()

            tok_res = self._evaluate_over_tokens(
                eval_token_perturbation_batch,
                num_tokens=num_tokens,
                model=wrapper,
                inputs=inputs,
                target_ids=target_ids,
                token_attribution=tok_attr,
                perturbation_steps=self.steps,
                pad_token_id=self.pad_token_id,
                special_token_ids=self.special_token_ids,
                semantic_mask=semantic_mask,
                filter_keywords=self.filter_keywords,
            )

            results["time_tok_pert"] = time.perf_counter() - start_time
            results.update(self._format_results("tok", tok_res, required_metrics=required_metrics))

        # --- C. Multimodal Synergy ---
        if pixel_attr is not None and tok_attr is not None:
            start_time = time.perf_counter()

            syn_res = self._evaluate_over_tokens(
                eval_multimodal_synergy_batch,
                num_tokens=num_tokens,
                model=wrapper,
                inputs=inputs,
                target_ids=target_ids,
                token_attribution=tok_attr,
                pixel_attribution=pixel_attr,
                perturbation_steps=self.steps,
                pad_token_id=self.pad_token_id,
                special_token_ids=self.special_token_ids,
                semantic_mask=semantic_mask,
                mask_value=self.mask_value,
                blur_baseline=blur_baseline,
                filter_keywords=self.filter_keywords,
            )

            results["time_syn_pert"] = time.perf_counter() - start_time
            results.update(self._format_results("syn", syn_res, required_metrics=required_metrics))

        return results

    def _format_results(
        self,
        prefix: str,
        raw_metrics: dict[str, Any],
        required_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Flattens arrays and adds modality prefixes."""
        if required_metrics is None:
            required_metrics = ["auc"]

        formatted = {}
        for key, val in raw_metrics.items():
            if not any(substring in key for substring in required_metrics):
                continue

            new_key = f"{prefix}_{key}"

            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()

            if isinstance(val, np.ndarray) and val.size == 1:
                formatted[new_key] = float(val.item())
            elif isinstance(val, np.ndarray) and "curve" in new_key:
                formatted[new_key] = safe_flatten_to_list(val)
            elif isinstance(val, (float, int)):
                formatted[new_key] = val
            else:
                formatted[new_key] = val

        return formatted


@torch.no_grad()
def eval_image_perturbation_batch(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    target_ids: Tensor,  # (B, L_label) - The expected answer tokens
    pixel_attribution: Tensor,  # (B, H, W) or (B, num_patches)
    perturbation_steps: Sequence[float],
    mask_value: float = 0.0,
    descending: bool = True,  # True = Deletion (remove important first), False = (remove important last)
    filter_keywords: bool = True,  # If True, only tracks tokens that are "visually dependent"
    blur_baseline: Tensor | None = None,  # Optional blurred image for keyword filtering
) -> dict[str, Any]:
    """
    Batch-level image perturbation evaluation specifically for VLMs.

    Adapts the logic of 'metric()' and 'score_output()' into a batched efficient format.
    """
    device = model.device
    pixel_values = inputs["pixel_values"].to(device)
    pixel_attribution = pixel_attribution.to(device)
    input_ids = inputs["input_ids"].to(device)
    target_ids = target_ids.to(device)

    # --- SAFELY ADD BATCH DIMENSION BASED ON ARCHITECTURE ---
    model_type = getattr(model.model.config, "model_type", "").lower()

    if "internvl" in model_type:
        # InternVL expects 5D: (Batch, num_tiles, C, H, W)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)

    elif "qwen" in model_type:
        # QwenVL expects 3D: (Batch, num_patches, patch_dim)
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)

    else:
        # Fallback for Standard VLMs like LLaVA
        # LLaVA expects 4D: (Batch, C, H, W)
        # Processors usually return 4D, so we ONLY unsqueeze if it's oddly 3D
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

    # ---------- normalize shapes & define feature/position dims ----------
    # ndim = pixel_values.ndim
    origin_shape = pixel_values.shape

    # Setup Baselines & Flattening
    feat, num_pixels = _reshape_pixels_faithfulness(
        pixel_values=pixel_values, origin_shape=origin_shape, model_type=model_type
    )

    # Baseline image (blur)
    if blur_baseline is None:
        # blur_baseline = torch.full_like(pixel_values, mask_value).to(device)
        blur_baseline = make_blur_baseline(pixel_values=pixel_values, model_type=model_type)
    feat_baseline = blur_baseline.clone().reshape(feat.shape)

    # Flatten Attribution Map
    B, *_ = origin_shape
    if pixel_attribution.ndim == 4 or pixel_attribution.ndim == 3:  # INTERNVL: (B, num_tiles, H, W)
        sal_flat = pixel_attribution.reshape(B, -1)
    elif pixel_attribution.ndim == 2:  # QWENVL: (B, num_patches)
        sal_flat = pixel_attribution
    else:
        raise ValueError("pixel_attribution must be 2D, 3D, or 4D.")

    # 1. Prepare Target Positions (The "visual keywords" logic)
    # We need to decide WHICH tokens in the target_ids we care about.
    # Default: Track all tokens in the target.
    # Smart: Track only tokens that drop in probability when image is blurred (find_keywords logic).

    if filter_keywords and blur_baseline is not None:
        target_positions = get_most_important_tokens_pixel(
            model,
            inputs=inputs,
            input_ids=input_ids,
            target_ids=target_ids,
            pixel_values=pixel_values,
            blur_baseline=blur_baseline,
        )
    else:
        # Default: Use all target tokens
        seq_len = target_ids.shape[1]
        default_indices = torch.arange(seq_len, device=device)
        target_positions = [default_indices for _ in range(B)]

    # Calculate Baseline Scores
    baseline_scores = score_output(
        model,
        inputs=inputs,
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()  # (B,)

    # If we have a blur baseline, we use its score for normalization (0.0 point)
    # If not, we use the score of a fully masked image (calculated later or assumed)
    # if blur_baseline is not None:
    blur_scores = score_output(
        model,
        inputs=inputs,
        input_ids=input_ids,
        pixel_values=blur_baseline,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()  # (B,)

    normalizer = np.maximum(np.abs(baseline_scores - blur_scores), 0.05)

    # else:
    #     # Temporary fallback if no blur image provided: assume 0.0 or calculate on fully masked
    #     blur_scores = np.zeros_like(baseline_scores) # Placeholder

    # 4. Perturbation Loop
    S = len(perturbation_steps)
    del_scores_perturb = np.zeros((S, B), dtype=np.float32)
    ins_scores_perturb = np.zeros((S, B), dtype=np.float32)
    normalized_del_scores = np.zeros((S, B), dtype=np.float32)
    normalized_ins_scores = np.zeros((S, B), dtype=np.float32)

    num_feats = feat.shape[1]  # C or patch_dim

    for i, step in enumerate(perturbation_steps):
        k = int(round(step * num_pixels))

        if k == 0:
            del_scores_perturb[i] = baseline_scores
            ins_scores_perturb[i] = blur_scores
            normalized_del_scores[i] = (del_scores_perturb[i] - blur_scores) / normalizer
            normalized_ins_scores[i] = (ins_scores_perturb[i] - blur_scores) / normalizer
            continue

        if k > 0:
            k = max(1, min(k, num_pixels))

        # Descending = True -> Remove highest saliency first (Deletion)
        _, flat_idx = torch.topk(sal_flat, k, dim=-1, largest=descending)  # (B, k)

        flat_idx_expanded = flat_idx.unsqueeze(1).expand(B, num_feats, k)  # (B, num_feats, k)

        # ----------------------------
        # Deletion
        # ----------------------------

        # Create Perturbed Batch
        feat_pert = feat.clone()
        # mask_src = torch.full_like(feat_pert[:, :, :k], mask_value)
        mask_src = feat_baseline.gather(dim=2, index=flat_idx_expanded)  # (B, num_feats, k)

        # Apply mask
        feat_pert.scatter_(dim=2, index=flat_idx_expanded, src=mask_src)  # (B, num_feats, k)

        # Reshape back to original pixel_values
        del_pixels = _reshape_pixels_back_faithfulness(
            feat_pert=feat_pert, origin_shape=origin_shape, model_type=model_type
        )

        # ----------------------------
        # Insertion
        # ----------------------------
        # Create Perturbed Batch
        feat_pert = feat_baseline.clone()
        # flat_idx_expanded = flat_idx.unsqueeze(1).expand(B,
        #                                                  num_feats,
        #                                                  k
        #                                                 ) # (B, num_feats, k)
        # mask_src = torch.full_like(feat_pert[:, :, :k], mask_value)
        mask_src = feat.gather(dim=2, index=flat_idx_expanded)  # (B, C, k)

        # Apply mask
        feat_pert.scatter_(dim=2, index=flat_idx_expanded, src=mask_src)  # (B, num_feats, k)

        # Reshape back to original pixel_values
        ins_pixels = _reshape_pixels_back_faithfulness(
            feat_pert=feat_pert, origin_shape=origin_shape, model_type=model_type
        )

        # ------------- Scoring --------------
        # Compute Deletion Scores
        current_del_scores = score_output(
            model,
            inputs=inputs,
            input_ids=input_ids,
            pixel_values=del_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()  # (B,)
        del_scores_perturb[i] = current_del_scores

        # Normalize: (Current - Blur) / (Original - Blur)
        # This matches: outputs = (outputs-blur_scores) / (og_scores-blur_scores)
        # norm_score = (current_del_scores - blur_scores) / (baseline_scores - blur_scores + 1e-9)
        norm_score = (current_del_scores - blur_scores) / normalizer
        normalized_del_scores[i] = np.clip(norm_score, 0.0, 1.0)

        # Compute Insertion Scores
        current_ins_scores = score_output(
            model,
            inputs=inputs,
            input_ids=input_ids,
            pixel_values=ins_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()  # (B,)
        ins_scores_perturb[i] = current_ins_scores

        # Normalize: (Current - Blur) / (Original - Blur)
        # norm_score = (current_ins_scores - blur_scores) / (baseline_scores - blur_scores + 1e-9)
        norm_score = (current_ins_scores - blur_scores) / normalizer
        normalized_ins_scores[i] = np.clip(norm_score, 0.0, 1.0)

    # Compute AUC scores
    norm_auc_del = np.trapezoid(normalized_del_scores, x=perturbation_steps, axis=0)
    norm_auc_ins = np.trapezoid(normalized_ins_scores, x=perturbation_steps, axis=0)
    auc_del = np.trapezoid(del_scores_perturb, x=perturbation_steps, axis=0)
    auc_ins = np.trapezoid(ins_scores_perturb, x=perturbation_steps, axis=0)

    return {
        "baseline_scores": baseline_scores,  # Raw scores of original image
        "blur_scores": blur_scores,  # Raw scores of baseline (blur) image
        "raw_curve_del": del_scores_perturb,  # (S, B) Raw del_scores at each step
        "normalized_curve_del": normalized_del_scores,  # (S, B) Normalized 0-1 scores (AUC ready)
        "raw_curve_ins": ins_scores_perturb,  # (S, B) Raw ins_scores at each step
        "normalized_curve_ins": normalized_ins_scores,  # (S, B) Normalized 0-1 scores (AUC ready)
        "norm_auc_del": norm_auc_del,
        "norm_auc_ins": norm_auc_ins,
        "auc_del": auc_del,
        "auc_ins": auc_ins,
    }


@torch.no_grad()
def eval_token_perturbation_batch(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    target_ids: Tensor,  # (B, L_label) - The answer
    token_attribution: Tensor,  # (B, L_prompt) - Importance of the prompt tokens
    perturbation_steps: Sequence[float],
    pad_token_id: int,  # The token ID used to mask text (e.g. tokenizer.pad_token_id)
    special_token_ids: list[int],
    semantic_mask: Tensor | None = None,
    descending: bool = True,  # True = Deletion (remove important first), False = (remove important last)
    filter_keywords: bool = True,
    # pixel_values are passed inside inputs or separately depending on your wrapper
) -> dict[str, Any]:

    device = model.device
    input_ids = inputs["input_ids"].to(device)
    target_ids = target_ids.to(device)
    token_attribution = token_attribution.to(device)
    pixel_values = inputs["pixel_values"].to(device)

    # Ensure attribution matches prompt length
    if token_attribution.shape[-1] != input_ids.shape[-1]:
        raise ValueError(f"Attribution length {token_attribution.shape} != Input length {input_ids.shape}")

    B, seq_len = input_ids.shape

    # --- Identify "Valid" Text Tokens ---
    # Start by only allowing perturbation where semantic_mask is True
    if semantic_mask is not None:
        if semantic_mask.ndim == 1:
            semantic_mask = semantic_mask.unsqueeze(0)
        valid_mask = semantic_mask.to(device).clone()
    else:
        # Fallback if no mask is provided: assume all tokens are valid
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)

    if special_token_ids is not None:
        for skip_id in special_token_ids:
            # Mark positions containing visual tokens as False
            valid_mask &= input_ids != skip_id

    # Count how many actual text tokens we have per batch
    # We take the min across batch to ensure consistent step sizes, or average.
    # For safety, let's assume batch has roughly same text length or take the first.
    # A better way is to calculate K per sample, but let's stick to a batch approximation.
    num_valid_tokens = valid_mask.sum(dim=1).min().item()

    # --- Mask Attribution Scores ---
    # We want to ensure visual tokens are NEVER picked as "most important".
    # We set their attribution to -infinity.

    masked_attribution = token_attribution.clone()
    # Apply huge negative value where mask is False (Visual tokens)
    masked_attribution[~valid_mask] = -float("inf")

    # --- Baselines ---
    # Baseline Input: Text is masked (PAD), but Visual tokens remain ORIGINAL.
    # This is crucial: We want to see if the TEXT matters, assuming the image is visible.
    baseline_input_ids = input_ids.clone()

    # Only mask the valid text tokens with PAD
    # We use valid_mask to select where to put the pads
    baseline_input_ids[valid_mask] = pad_token_id

    target_positions = []
    if filter_keywords:
        # We need a wrapper to call get_most_important_tokens with the text baseline
        # Temporarily swap input_ids in the inputs dict?
        # Easier to call the logic directly:

        # Calculate keywords based on: "Does masking the prompt destroy the answer?"
        target_positions = get_most_important_tokens_token(model, inputs, input_ids, baseline_input_ids, target_ids)
    else:
        # Default: Use all target tokens
        seq_len = target_ids.shape[1]
        default_indices = torch.arange(seq_len, device=device)
        target_positions = [default_indices for _ in range(B)]

    # 3. Compute Baseline Scores
    # Score with Full Prompt
    baseline_scores = score_output(
        model,
        inputs=inputs,
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()

    # Score with Empty Prompt (Baseline)
    blur_scores = score_output(
        model,
        inputs=inputs,
        input_ids=baseline_input_ids,
        pixel_values=pixel_values,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()

    normalizer = np.maximum(np.abs(baseline_scores - blur_scores), 0.05)

    # 4. Perturbation Loop
    S = len(perturbation_steps)
    del_curve = np.zeros((S, B), dtype=np.float32)
    ins_curve = np.zeros((S, B), dtype=np.float32)
    norm_del_curve = np.zeros((S, B), dtype=np.float32)
    norm_ins_curve = np.zeros((S, B), dtype=np.float32)

    for i, step in enumerate(perturbation_steps):
        # Calculate how many tokens to mask
        # k = int(round(step * seq_len))
        k = int(round(step * num_valid_tokens))

        if k == 0:
            del_curve[i] = baseline_scores
            ins_curve[i] = blur_scores
            norm_del_curve[i] = (del_curve[i] - blur_scores) / normalizer
            norm_ins_curve[i] = (ins_curve[i] - blur_scores) / normalizer
            continue

        # k = min(k, seq_len)
        k = int(min(k, num_valid_tokens))

        # Identify Top-K Tokens
        # We always want the "Most Important" tokens
        _, top_indices = torch.topk(masked_attribution, k, dim=-1, largest=descending)  # (B, k)

        # ------------------------------
        # Deletion: Original -> Pad
        # ------------------------------
        # Start with Original
        curr_input_del = input_ids.clone()
        # Create a source of PAD tokens
        pad_src = torch.full_like(top_indices, pad_token_id)
        # Scatter PADs into the Top-K positions
        curr_input_del.scatter_(dim=1, index=top_indices, src=pad_src)

        # ------------------------------
        # Insertion: Pad -> Original
        # ------------------------------
        # Start with Baseline (All Pad)
        curr_input_ins = baseline_input_ids.clone()
        # Gather Original tokens
        orig_tokens = input_ids.gather(dim=1, index=top_indices)
        # Scatter Original tokens into the Pad sequence
        curr_input_ins.scatter_(dim=1, index=top_indices, src=orig_tokens)

        # ------------ Scoring ------------
        # Deletion
        s_del = score_output(
            model,
            inputs=inputs,
            input_ids=curr_input_del,
            pixel_values=pixel_values,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()
        del_curve[i] = s_del
        # norm_del_curve[i] = (s_del - blur_scores) / (baseline_scores - blur_scores + 1e-9)
        norm_del = (s_del - blur_scores) / normalizer
        norm_del_curve[i] = np.clip(norm_del, 0.0, 1.0)

        # Insertion
        s_ins = score_output(
            model,
            inputs=inputs,
            input_ids=curr_input_ins,
            pixel_values=pixel_values,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()
        ins_curve[i] = s_ins
        # norm_ins_curve[i] = (s_ins - blur_scores) / (baseline_scores - blur_scores + 1e-9)
        norm_ins = (s_ins - blur_scores) / normalizer
        norm_ins_curve[i] = np.clip(norm_ins, 0.0, 1.0)

    # Compute AUC scores
    norm_auc_del = np.trapezoid(norm_del_curve, x=perturbation_steps, axis=0)
    norm_auc_ins = np.trapezoid(norm_ins_curve, x=perturbation_steps, axis=0)

    auc_del = np.trapezoid(del_curve, x=perturbation_steps, axis=0)
    auc_ins = np.trapezoid(ins_curve, x=perturbation_steps, axis=0)

    return {
        "baseline_scores": baseline_scores,
        "blur_scores": blur_scores,
        "raw_del_curve": del_curve,
        "raw_ins_curve": ins_curve,
        "normalized_del_curve": norm_del_curve,
        "normalized_ins_curve": norm_ins_curve,
        "norm_auc_del": norm_auc_del,
        "norm_auc_ins": norm_auc_ins,
        "auc_del": auc_del,
        "auc_ins": auc_ins,
    }


@torch.no_grad()
def eval_multimodal_synergy_batch(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    target_ids: Tensor,
    pixel_attribution: Tensor,  # (B, H, W)
    token_attribution: Tensor,  # (B, L_prompt)
    perturbation_steps: Sequence[float],
    pad_token_id: int,  # Text Baseline
    special_token_ids: list[int],  # For filtering text tokens
    semantic_mask: Tensor | None = None,
    blur_baseline: Tensor | None = None,  # Image Baseline (same shape as pixel_values)
    mask_value: float = 0.0,
    descending: bool = True,  # True = "Insertion" style (Start from 0, add Important)
    filter_keywords: bool = True,
) -> dict[str, Any]:
    """
    Computes the Synergy between Image and Text attributions.
    Formula: P(Img, Txt) - (P(Img, 0) + P(0, Txt))

    This is effectively a 'Double Insertion' metric.
    """
    device = model.device
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    pixel_attribution = pixel_attribution.to(device)
    token_attribution = token_attribution.to(device)
    target_ids = target_ids.to(device)

    # --- SAFELY ADD BATCH DIMENSION BASED ON ARCHITECTURE ---
    model_type = getattr(model.model.config, "model_type", "").lower()

    if "internvl" in model_type:
        # InternVL expects 5D: (Batch, num_tiles, C, H, W)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)

    elif "qwen" in model_type:
        # QwenVL expects 3D: (Batch, num_patches, patch_dim)
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)

    else:
        # Fallback for Standard VLMs like LLaVA
        # LLaVA expects 4D: (Batch, C, H, W)
        # Processors usually return 4D, so we ONLY unsqueeze if it's oddly 3D
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

    # ---------- normalize shapes & define feature/position dims for Image Inputs ----------
    # ndim = pixel_values.ndim
    origin_shape = pixel_values.shape

    model_type = model.model.config.model_type

    # # Setup Baselines & Flattening
    feat, num_pixels = _reshape_pixels_faithfulness(
        pixel_values=pixel_values, origin_shape=origin_shape, model_type=model_type
    )

    num_img_feat = feat.shape[1]  # C or patch_dim

    # Baseline image (blur)
    if blur_baseline is None:
        # blur_baseline = torch.full_like(pixel_values, mask_value).to(device)
        blur_baseline = make_blur_baseline(pixel_values=pixel_values, model_type=model_type)
    feat_baseline = blur_baseline.clone().reshape(feat.shape)

    # Flatten Attribution Map
    B, *_ = origin_shape
    if pixel_attribution.ndim == 4 or pixel_attribution.ndim == 3:  # INTERNVL: (B, num_tiles, H, W)
        sal_flat_img = pixel_attribution.reshape(B, -1)
    elif pixel_attribution.ndim == 2:  # QWENVL: (B, num_patches)
        sal_flat_img = pixel_attribution
    else:
        raise ValueError("pixel_attribution must be 2D, 3D, or 4D.")

    # --- 2. Setup Text Inputs ---
    # Start by only allowing perturbation where semantic_mask is True
    if semantic_mask is not None:
        if semantic_mask.ndim == 1:
            semantic_mask = semantic_mask.unsqueeze(0)
        valid_mask = semantic_mask.to(device).clone()
    else:
        # Fallback if no mask is provided: assume all tokens are valid
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)

    if special_token_ids is not None:
        for skip_id in special_token_ids:
            # Mark positions containing visual tokens as False
            valid_mask &= input_ids != skip_id

    # Count how many actual text tokens we have per batch
    num_valid_tokens = valid_mask.sum(dim=1).min().item()

    # --- Mask Attribution Scores ---
    masked_attribution = token_attribution.clone()
    # Apply huge negative value where mask is False (Visual tokens)
    masked_attribution[~valid_mask] = -float("inf")

    # --- Baselines ---
    baseline_input_ids = input_ids.clone()
    baseline_input_ids[valid_mask] = pad_token_id

    # --- 3. Baselines & Targets ---

    # Keyword filtering (Optional - calculate on Joint Original)
    if filter_keywords:
        # Use existing helper to find keywords on the FULL original input
        target_positions = get_most_important_tokens_multimodal(
            model,
            inputs,
            input_ids,
            baseline_input_ids,
            pixel_values,
            blur_baseline,
            target_ids,
        )
    else:
        # Default: Use all target tokens
        seq_len = target_ids.shape[1]
        default_indices = torch.arange(seq_len, device=device)
        target_positions = [default_indices for _ in range(B)]

    # Compute Global Baseline Score P(0, 0) and P(Img, Txt)
    # Needed for normalization if desired,

    # Image=Blur, Text=Pad
    zeros_scores = score_output(
        model,
        inputs=inputs,
        input_ids=baseline_input_ids,
        pixel_values=blur_baseline,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()  # (B,)

    # Image=Full, Text=Full
    full_scores = score_output(
        model,
        inputs=inputs,
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_ids=target_ids,
        positions=target_positions,
    ).numpy()  # (B,)

    # normalizer_ins = full_scores - zeros_scores
    # normalizer_del = full_scores - zeros_scores
    normalizer = np.maximum(np.abs(full_scores - zeros_scores), 0.05)

    # --- 4. Loop ---
    S = len(perturbation_steps)
    del_synergy_curve = np.zeros((S, B), dtype=np.float32)
    ins_synergy_curve = np.zeros((S, B), dtype=np.float32)
    del_norm_synergy_curve = np.zeros((S, B), dtype=np.float32)
    ins_norm_synergy_curve = np.zeros((S, B), dtype=np.float32)

    for i, step in enumerate(perturbation_steps):
        # --- A. Determine K for both modalities ---
        k_img = int(round(step * num_pixels))
        k_img = max(0, min(k_img, num_pixels))  # Allow 0

        k_txt = int(round(step * num_valid_tokens))
        k_txt = int(max(0, min(k_txt, num_valid_tokens)))  # Allow 0

        # Identify Top Pixels
        # descending=True means "most important first"
        # So we want Top K Largest values.
        _, top_img_idx = torch.topk(sal_flat_img, k_img, dim=-1, largest=descending)
        top_img_idx_exp = top_img_idx.unsqueeze(1).expand(B, num_img_feat, k_img)

        # Identify Top-K Tokens
        # We always want the "Most Important" tokens
        _, top_token_idx = torch.topk(masked_attribution, k_txt, dim=-1, largest=descending)  # (B, k)

        # ----------------------------
        # Deletion
        # ----------------------------

        # Construct "del_pixel" (Original Image - Top K Pixels)
        feat_pixels = feat.clone()
        pixels_orig = feat_baseline.gather(dim=2, index=top_img_idx_exp)

        # Apply mask on img
        feat_pixels.scatter_(dim=2, index=top_img_idx_exp, src=pixels_orig)

        # # Reshape back to original pixel_values
        del_pixels = _reshape_pixels_back_faithfulness(
            feat_pert=feat_pixels, origin_shape=origin_shape, model_type=model_type
        )

        # Construct "del_tokens" (Original Tokens - Top K Tokens)
        del_tokens = input_ids.clone()
        pad_src = torch.full_like(top_token_idx, pad_token_id)
        # Scatter PADs into the Top-K positions
        del_tokens.scatter_(dim=1, index=top_token_idx, src=pad_src)

        # ----------------------------
        # Insertion
        # ----------------------------

        # Construct "ins_pixel" (Blur Image + Top K Pixels)
        feat_pert = feat_baseline.clone()
        mask_src = feat.gather(dim=2, index=top_img_idx_exp)  # (B, C, k)

        # Apply mask
        feat_pert.scatter_(dim=2, index=top_img_idx_exp, src=mask_src)  # (B, num_feats, k)

        # Reshape back to original pixel_values
        ins_pixels = _reshape_pixels_back_faithfulness(
            feat_pert=feat_pert, origin_shape=origin_shape, model_type=model_type
        )

        # Construct "ins_tokens" (Pad Tokens + Top K Tokens)
        ins_tokens = baseline_input_ids.clone()
        # Gather Original tokens
        orig_tokens = input_ids.gather(dim=1, index=top_token_idx)
        # Scatter Original tokens into the Pad sequence
        ins_tokens.scatter_(dim=1, index=top_token_idx, src=orig_tokens)

        # --- Deletion Scoring ---

        # Joint: P(Img\Img_k, Txt\Txt_k)
        del_p_joint = score_output(
            model,
            inputs,
            input_ids=del_tokens,
            pixel_values=del_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # 2. Image Only: P(Img\Img_k, Txt)
        # Input: img_s_img, txt_baseline
        del_p_img_only = score_output(
            model,
            inputs,
            input_ids=input_ids,
            pixel_values=del_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # 3. Text Only: P(Img, Txt\Txt_k)
        # Input: img_baseline, txt_s_txt
        del_p_txt_only = score_output(
            model,
            inputs,
            input_ids=del_tokens,
            pixel_values=pixel_values,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # --- E. Calculate Synergy ---
        # Formula: - P(Img\Img_k, Txt\Txt_k) + P(Img\Img_k, Txt)
        #           + P(Img, Txt\Txt_k) - P(Img, Txt)
        del_synergy = del_p_joint - (del_p_img_only + del_p_txt_only - full_scores)
        del_synergy_curve[i] = del_synergy
        # del_norm_synergy_curve[i] = del_p_joint - (del_p_img_only + del_p_txt_only - zeros_scores)
        # del_norm_synergy_curve[i] /= (normalizer_del + 1e-9)
        # del_norm_synergy_curve[i] += 1
        del_norm = del_synergy / normalizer
        del_norm_synergy_curve[i] = np.clip(del_norm, 0.0, 1.0)

        # --- Insertion Scoring ---

        # Joint: P(Img_k, Txt_k)
        ins_p_joint = score_output(
            model,
            inputs,
            input_ids=ins_tokens,
            pixel_values=ins_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # 2. Image Only: P(Img_k, 0)
        # Input: img_s_img, txt_baseline
        ins_p_img_only = score_output(
            model,
            inputs,
            input_ids=baseline_input_ids,
            pixel_values=ins_pixels,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # 3. Text Only: P(0, Txt_k)
        # Input: img_baseline, txt_s_txt
        ins_p_txt_only = score_output(
            model,
            inputs,
            input_ids=ins_tokens,
            pixel_values=blur_baseline,
            output_ids=target_ids,
            positions=target_positions,
        ).numpy()

        # --- E. Calculate Synergy ---
        # Formula: P(Img_k, Txt_k) - P(Img_k, 0)
        #           - P(0, Txt_k) + P(0, 0)
        ins_synergy = ins_p_joint - (ins_p_img_only + ins_p_txt_only - zeros_scores)
        # ins_synergy = ins_p_joint - (ins_p_img_only + ins_p_txt_only)
        ins_synergy_curve[i] = ins_synergy
        # ins_norm_synergy_curve[i] = ins_synergy / (normalizer_ins + 1e-9)
        ins_norm = ins_synergy / normalizer
        ins_norm_synergy_curve[i] = np.clip(ins_norm, 0.0, 1.0)

    del_norm_syn_auc = np.trapezoid(del_norm_synergy_curve, x=perturbation_steps, axis=0)
    ins_norm_syn_auc = np.trapezoid(ins_norm_synergy_curve, x=perturbation_steps, axis=0)
    del_syn_auc = np.trapezoid(del_synergy_curve, x=perturbation_steps, axis=0)
    ins_syn_auc = np.trapezoid(ins_synergy_curve, x=perturbation_steps, axis=0)

    synergy_curve = (ins_synergy_curve + del_synergy_curve) / 2.0
    synergy_norm_curve = (ins_norm_synergy_curve + del_norm_synergy_curve) / 2.0
    synergy_norm_auc = np.trapezoid(synergy_norm_curve, x=perturbation_steps, axis=0)
    synergy_auc = np.trapezoid(synergy_curve, x=perturbation_steps, axis=0)

    return {
        "del_synergy_curve": del_synergy_curve,
        "ins_synergy_curve": ins_synergy_curve,
        "ins_norm_synergy_curve": ins_norm_synergy_curve,
        "del_norm_synergy_curve": del_norm_synergy_curve,
        "zeros_baseline": zeros_scores,
        "full_baseline": full_scores,
        "del_norm_auc": del_norm_syn_auc,
        "ins_norm_auc": ins_norm_syn_auc,
        "del_auc": del_syn_auc,
        "ins_auc": ins_syn_auc,
        "synergy_auc": synergy_auc,
        "synergy_norm_auc": synergy_norm_auc,
    }
