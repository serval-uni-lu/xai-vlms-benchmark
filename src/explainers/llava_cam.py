import contextlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from src.utils.xai_utils import align_llm_visuals_to_pixels


class LLaVACAMExplainer(BaseExplainer):
    def __init__(
        self,
        model_wrapper: BaseVLMWrapper,
        target_layer_name: str,
        num_samples: int = 10,
        noise_std: float = 0.1,
        token_wise=True,
    ):
        super().__init__(model_wrapper)

        self.target_layer_name = target_layer_name
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.token_wise = token_wise

    def save_feature_maps(self, module, input, output):
        """Hook to save the feature maps during forward pass."""
        self.feature_maps = output
        # output.retain_grad()

    def save_gradients(self, module, grad_input, grad_output):
        """Hook to save the gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        for name, module in self.wrapper.model.named_modules():
            if self.target_layer_name in name:
                module.register_forward_hook(self.save_feature_maps)
                module.register_backward_hook(self.save_gradients)

    def clear_hooks(self):
        self.feature_maps = None
        self.gradients = None

        torch.cuda.empty_cache()

    @contextlib.contextmanager
    def manage_explainability_state(self):
        """
        Temporarily patches the model and attaches hooks.
        Guarantees complete restoration upon exit.
        """

        # Register hooks to capture activations (forward) and gradients (backward)
        self.register_hooks()

        try:
            # Yield control back to the main function
            yield
        finally:
            # This runs NO MATTER WHAT (even if your math throws an error)
            self.clear_hooks()

    def compute_cam(self, mask):
        """Applies Grad-CAM channel-weighting to the selected tokens."""
        # Slice the sequence down to just the tokens we care about
        # Shape: [num_selected_tokens, Channels]

        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError("Gradients or Feature maps were dropped/not initialized")

        activations = self.feature_maps
        gradients = self.gradients

        # MULTI-GPU MASK DEVICE FIX
        mask = mask.to(activations.device)

        feats = activations[0, mask, :]
        grads = gradients[0, mask, :]

        # Global Average Pooling of the gradients across the tokens
        # Shape: [Channels]
        pooled_grads = grads.mean(dim=0)

        # Weight the feature maps
        # [num_selected_tokens, Channels] * [Channels]
        weighted_feats = feats * pooled_grads

        # Average across the channel dimension to get a single score per token
        # Shape: [num_selected_tokens]
        cam = weighted_feats.sum(dim=-1)

        # Grad-CAM requires a ReLU to keep only positive contributions
        cam = nn.functional.relu(cam)

        # # Normalize to [0, 1]
        # if cam.max() > 0:
        #     cam = cam / cam.max()

        return cam.detach().cpu()

    def _add_noise(self, image, noise_std):
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.copy().convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        img_arr = np.array(pil_img).astype(np.float32)
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_std * 255, img_arr.shape)
        noisy_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

        noisy_pil = Image.fromarray(noisy_arr)

        return noisy_pil

    def _attribute(
        self, image, text, target_indices: int | list[int] | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Grad-CAM
        if self.num_samples <= 1:
            smooth_token_cam, smooth_pixel_cam = self.llava_cam(
                image=image, text=text, target_indices=target_indices, **kwargs
            )
            return smooth_token_cam, smooth_pixel_cam

        # SmoothGRAD-CAM
        noisy_img = self._add_noise(image, noise_std=self.noise_std)
        smooth_token_cam, smooth_pixel_cam = self.llava_cam(
            image=noisy_img, text=text, target_indices=target_indices, **kwargs
        )

        for _ in range(self.num_samples - 1):
            noisy_img = self._add_noise(image, noise_std=self.noise_std)
            t_cam, p_cam = self.llava_cam(image=noisy_img, text=text, target_indices=target_indices, **kwargs)

            smooth_token_cam += t_cam
            smooth_pixel_cam += p_cam

        # Average
        smooth_token_cam /= self.num_samples
        smooth_pixel_cam /= self.num_samples

        return smooth_token_cam.float(), smooth_pixel_cam.float()

    def llava_cam(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates raw attributions dynamically for specified targets.
        """
        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(
                inputs=inputs,
                return_logits=False,
                **kwargs,
            )
        full_ids = pred_results["full_ids"].to(self.device)

        # Define the indices of the answers tokens and visual tokens
        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]
        gen_len = t_end - t_start

        # --- DYNAMIC INDICES RESOLUTION ---
        if target_indices is None:
            indices_to_compute = list(range(gen_len))
        elif isinstance(target_indices, int):
            indices_to_compute = [target_indices]
        else:
            indices_to_compute = target_indices

        # Safety check
        indices_to_compute = [idx for idx in indices_to_compute if idx < gen_len]

        inputs["input_ids"] = full_ids.clone()  # (batch, seq_len)
        if inputs["input_ids"].ndim == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # --- MULTIMODAL ROPE FIX FOR QWEN ---
        # If the model requires mm_token_type_ids, we must expand it to match the new input_ids length.
        # Since all generated tokens are text, we pad it with the text type (0).
        if "mm_token_type_ids" in inputs:
            orig_mm_types = inputs["mm_token_type_ids"]
            
            # Create a tensor of 0s (text type) for the newly generated answer tokens
            pad_types = torch.zeros(
                (orig_mm_types.shape[0], gen_len), 
                dtype=orig_mm_types.dtype, 
                device=orig_mm_types.device
            )
            
            # Concatenate them together so it matches the length of the full_ids
            inputs["mm_token_type_ids"] = torch.cat([orig_mm_types, pad_types], dim=1)
            
        # Optional: Also catch position_ids if they were explicitly passed by the processor
        if "position_ids" in inputs:
            # We delete it so Qwen is forced to dynamically recalculate the correct 
            # position IDs using the newly padded mm_token_type_ids
            del inputs["position_ids"]

        image_token_id = self.wrapper.model.config.image_token_id

        # Boolean Masks (flattened to 1D)
        full_ids_1d = inputs["input_ids"].squeeze()
        prompt_mask = torch.arange(full_ids_1d.size(-1), device=full_ids_1d.device) < t_start

        is_image_mask = full_ids_1d == image_token_id
        # is_text_mask = ~is_image_mask

        final_text_mask = prompt_mask
        final_image_mask = is_image_mask & prompt_mask

        with self.manage_explainability_state():
            outputs = self.wrapper.model(
                **inputs,
                use_cache=False,  # Crucial for backward hooks
            )
            logits = outputs.logits[:, t_start - 1 : t_end - 1, :]  # (1, num_ans_tokens, vocab_size)

            new_ids = full_ids[t_start:t_end].unsqueeze(0).unsqueeze(-1)
            target_logits = logits.gather(dim=-1, index=new_ids).squeeze(-1)  # (1, num_ans_tokens)

            if self.token_wise:
                token_attributions = []
                pixel_attributions = []

                # --- TARGETED LOOP: Only backward pass on requested indices ---
                for i in indices_to_compute:
                    self.wrapper.model.zero_grad()
                    target_logits[:, i].backward(retain_graph=True)

                    if self.gradients is None:
                        raise RuntimeError(
                            f"Gradients were dropped at iteration {i}. Check layer name: {self.target_layer_name}"
                        )

                    # Compute CAM for text and image
                    tok_attr = self.compute_cam(final_text_mask)
                    pix_attr = self.compute_cam(final_image_mask)

                    self.gradients = None

                    token_attributions.append(tok_attr)
                    pixel_attributions.append(pix_attr)

                token_attribution = torch.stack(token_attributions, dim=0)  # [num_targets, num_text_tokens]
                pixel_attribution = torch.stack(pixel_attributions, dim=0)  # [num_targets, num_image_tokens]

            else:
                # --- AGGREGATED TARGETS: Only sum the requested indices ---
                answer_score = target_logits[0, indices_to_compute].sum()
                self.wrapper.model.zero_grad()
                answer_score.backward(retain_graph=True)

                token_attribution = self.compute_cam(final_text_mask).unsqueeze(0)
                pixel_attribution = self.compute_cam(final_image_mask).unsqueeze(0)

        # Map back to 2D/3D pixel space
        pixel_attribution = align_llm_visuals_to_pixels(pixel_attribution, inputs, config=self.wrapper.model.config)

        return token_attribution, pixel_attribution
