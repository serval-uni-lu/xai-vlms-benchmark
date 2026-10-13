import contextlib

import torch
import torch.nn.functional as F

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from src.utils.xai_utils import align_llm_visuals_to_pixels


class AtManExplainer(BaseExplainer):
    def __init__(
        self,
        model_wrapper: BaseVLMWrapper,
        batch_size: int = 16,
        conceptual_threshold: float | None = None,
    ):
        """
        Args:
            model_wrapper: The VLM wrapper.
            batch_size: How many perturbed forward passes to run at once.
            conceptual_threshold: If set (e.g., 0.6), suppresses all tokens conceptually
                                  similar to the target token. If None, does standard AtMan.
        """
        super().__init__(model_wrapper)
        self.batch_size = batch_size
        self.conceptual_threshold = conceptual_threshold

        # State variables for the hooks
        self.hooks = []
        self.current_suppress_mask = None

    def _get_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculates the cosine similarity matrix for a sequence of embeddings."""
        emb = embeddings.squeeze(0).float()
        emb_norm = F.normalize(emb, p=2, dim=-1, eps=1e-8)
        sim_matrix = torch.mm(emb_norm, emb_norm.transpose(0, 1))
        return sim_matrix.clip(-1.0, 1.0)

    def _interceptor_hook(self, module, args, kwargs):
        """
        Pre-hook to actively suppress tokens right before attention math.
        Intercepts the attention_mask and forces target tokens to -inf.
        """
        if self.current_suppress_mask is None:
            return args, kwargs

        attn_mask = None
        mask_in_kwargs = False
        mask_arg_idx = -1

        # 1. Safely locate the attention_mask in kwargs or args
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            attn_mask = kwargs["attention_mask"]
            mask_in_kwargs = True
        elif len(args) > 1 and args[1] is not None:
            attn_mask = args[1]
            mask_arg_idx = 1

        if attn_mask is not None:
            modified_mask = attn_mask.clone()

            # Expand our 2D boolean mask to match HF's internal 4D extended mask
            # [batch, seq] -> [batch, 1, 1, seq]
            b_mask = self.current_suppress_mask.unsqueeze(1).unsqueeze(1).to(modified_mask.device)

            # Softmax(-inf) = 0, effectively deleting the token from attention
            dtype_min = torch.finfo(modified_mask.dtype).min
            modified_mask = modified_mask.masked_fill(b_mask, dtype_min)

            # Inject the modified mask back into the model's forward pass
            if mask_in_kwargs:
                kwargs["attention_mask"] = modified_mask
            else:
                args_list = list(args)
                args_list[mask_arg_idx] = modified_mask
                args = tuple(args_list)

        return args, kwargs

    def register_hooks(self):
        """Attaches the interceptor pre-hook specifically to the LLM layers."""
        self.clear_hooks()
        for _, module in self.wrapper.model.named_modules():
            # Target only the LLM layers (skips the Vision Encoder)
            if self.wrapper.llm_module_name and self.wrapper.llm_module_name in str(type(module)):
                # if hasattr(module, "self_attn"): # Ensure it's an attention block
                # register_forward_pre_hook with kwargs requires PyTorch >= 2.0
                self.hooks.append(module.register_forward_pre_hook(self._interceptor_hook, with_kwargs=True))

    def clear_hooks(self):
        """Removes all hooks and resets the state."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.current_suppress_mask = None
        torch.cuda.empty_cache()

    @contextlib.contextmanager
    def manage_explainability_state(self):
        """Context manager to safely attach and detach hooks."""
        self.register_hooks()
        try:
            yield
        finally:
            self.clear_hooks()

    def _attribute(
        self, image, text: str, target_indices: int | list[int] | None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs=inputs, return_logits=False, **kwargs)

        full_ids = pred_results["full_ids"]
        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]
        gen_len = t_end - t_start

        base_input_ids = full_ids.clone().unsqueeze(0)
        base_attention_mask = torch.ones_like(base_input_ids)

        # Isolate visual arguments (pixel_values, etc.) so we don't overwrite them
        visual_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}

        # 1. RUN CLEAN BASELINE PASS (No hooks active here)
        with torch.no_grad():
            clean_outputs = self.wrapper.model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                output_hidden_states=True,
                **visual_kwargs,
            )

            clean_logits = clean_outputs.logits[:, t_start - 1 : t_end - 1, :]
            target_ids = full_ids[t_start:t_end].unsqueeze(0).unsqueeze(-1)

            clean_log_probs = F.log_softmax(clean_logits, dim=-1)
            clean_target_log_probs = clean_log_probs.gather(dim=-1, index=target_ids).squeeze(-1)

            if self.conceptual_threshold is not None:
                # Layer 0 hidden states for pure visual/text semantic similarity
                base_embeddings = clean_outputs.hidden_states[0]
                similarity_matrix = self._get_similarity_matrix(base_embeddings)

        # 2. IDENTIFY TOKENS TO PERTURB
        image_token_id = self.wrapper.model.config.image_token_id
        full_ids_1d = full_ids.squeeze()

        prompt_mask = torch.arange(full_ids_1d.size(-1), device=full_ids_1d.device) < t_start
        is_image_mask = full_ids_1d == image_token_id
        is_text_mask = ~is_image_mask

        text_indices_to_mask = torch.where(is_text_mask & prompt_mask)[0]
        image_indices_to_mask = torch.where(is_image_mask & prompt_mask)[0]
        all_indices_to_mask = torch.cat([text_indices_to_mask, image_indices_to_mask])

        num_perturbations = len(all_indices_to_mask)
        attribution_matrix = torch.zeros((gen_len, base_input_ids.shape[1]), device=base_input_ids.device)

        # 3. RUN PERTURBED FORWARD PASSES (With hooks active)
        with self.manage_explainability_state():
            for i in range(0, num_perturbations, self.batch_size):
                batch_indices = all_indices_to_mask[i : i + self.batch_size]
                current_b_size = len(batch_indices)

                b_input_ids = base_input_ids.expand(current_b_size, -1)

                # Pass a completely clean mask externally to keep RoPE coordinates intact
                b_clean_attention_mask = base_attention_mask.expand(current_b_size, -1)

                # Batch the visual arguments safely to prevent out-of-bounds indexing
                b_visual_kwargs = {}
                for k, v in visual_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        repeats = [1] * v.dim()
                        repeats[0] = current_b_size
                        b_visual_kwargs[k] = v.repeat(*repeats)
                    elif isinstance(v, list):
                        b_visual_kwargs[k] = v * current_b_size
                    else:
                        b_visual_kwargs[k] = v

                # Create the boolean mask that our interceptor hook will use
                suppress_bool_mask = torch.zeros(
                    (current_b_size, base_input_ids.shape[1]),
                    dtype=torch.bool,
                    device=base_input_ids.device,
                )

                for batch_idx, token_idx_to_suppress in enumerate(batch_indices):
                    if self.conceptual_threshold is not None:
                        sim_scores = similarity_matrix[token_idx_to_suppress]
                        similar_indices = torch.where(sim_scores >= self.conceptual_threshold)[0]
                        suppress_bool_mask[batch_idx, similar_indices] = True
                    else:
                        suppress_bool_mask[batch_idx, token_idx_to_suppress] = True

                # Broadcast the mask to the class state so the hook can read it
                self.current_suppress_mask = suppress_bool_mask

                # Forward pass: the model thinks it's normal, but the hook sabotages the internal layers
                with torch.no_grad():
                    perturbed_outputs = self.wrapper.model(
                        input_ids=b_input_ids,
                        attention_mask=b_clean_attention_mask,
                        **b_visual_kwargs,
                    )

                perturbed_logits = perturbed_outputs.logits[:, t_start - 1 : t_end - 1, :]
                perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
                b_target_ids = target_ids.expand(current_b_size, -1, -1)
                perturbed_target_log_probs = perturbed_log_probs.gather(dim=-1, index=b_target_ids).squeeze(-1)

                # 4. CALCULATE RELEVANCE
                prob_drops = clean_target_log_probs.expand(current_b_size, -1) - perturbed_target_log_probs
                prob_drops = torch.clamp(prob_drops, min=0.0)

                for batch_idx, token_idx_to_suppress in enumerate(batch_indices):
                    attribution_matrix[:, token_idx_to_suppress] = prob_drops[batch_idx]

        # 5. FORMAT AND RETURN
        # final_text_mask = is_text_mask & prompt_mask
        final_image_mask = is_image_mask & prompt_mask

        token_attribution = attribution_matrix[:, prompt_mask]
        raw_pixel_attribution = attribution_matrix[:, final_image_mask]

        pixel_attribution = align_llm_visuals_to_pixels(raw_pixel_attribution, inputs, config=self.wrapper.model.config)

        return token_attribution.detach().cpu(), pixel_attribution.detach().cpu()
