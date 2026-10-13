import torch

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper


class RandomExplainer(BaseExplainer):
    """
    A baseline explainer that returns random attributions.
    Used to validate that faithfulness metrics correctly penalize random noise.
    Compatible with Standard 4D, QwenVL 3D (patches), and InternVL 5D (tiles).
    """

    def __init__(self, model_wrapper: "BaseVLMWrapper"):
        super().__init__(model_wrapper)

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        seed: int = 42,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates random attention maps for Image and Text.
        """
        # 1. Prepare Inputs using the wrapper's processor
        inputs = self.wrapper.get_inputs(image, text)

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        # --- SAFELY ADD BATCH DIMENSION BASED ON ARCHITECTURE ---
        model_type = getattr(self.wrapper.model.config, "model_type", "").lower()

        if "internvl" in model_type:
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(0)
        elif "qwen" in model_type:
            if pixel_values.ndim == 2:
                pixel_values = pixel_values.unsqueeze(0)
        else:
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)

        # Retrieve or generate predictions
        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(
                inputs=inputs,
                return_logits=False,
                **kwargs,
            )

        new_ids = pred_results["new_ids"]
        if new_ids.dim() > 1:
            new_ids = new_ids[0]
        seq_len_generated = len(new_ids)

        # --- DYNAMIC INDICES RESOLUTION ---
        if target_indices is None:
            indices_to_compute = list(range(seq_len_generated))
        elif isinstance(target_indices, int):
            indices_to_compute = [target_indices]
        else:
            indices_to_compute = target_indices

        indices_to_compute = [idx for idx in indices_to_compute if idx < seq_len_generated]
        num_targets = len(indices_to_compute)

        # 2. Set Seed for Reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # 3. Generate Random Token Attributions
        # We only generate noise for the specific number of target tokens!
        prompt_len = input_ids.shape[1]
        token_attributions = torch.rand((num_targets, prompt_len), device=self.device, generator=generator)

        # 4. Generate Random Pixel Attributions based on model architecture
        # ndim = pixel_values.ndim

        if "internvl" in model_type:
            # --- InternVL ---
            _, num_tiles, _, h, w = pixel_values.shape
            pixel_attributions = torch.rand((num_targets, num_tiles, h, w), device=self.device, generator=generator)

        elif "llava" in model_type:
            # --- Standard CNN/ViT (LLaVA) ---
            _, _, h, w = pixel_values.shape
            pixel_attributions = torch.rand((num_targets, h, w), device=self.device, generator=generator)

        elif "qwen" in model_type:
            # --- Qwen2-VL ---
            _, num_patches, _ = pixel_values.shape

            if "image_grid_thw" in inputs:
                grid_thw = inputs["image_grid_thw"][0].cpu().numpy().tolist()
                h, w = grid_thw[1], grid_thw[2]
                pixel_attributions = torch.rand((num_targets, h * w), device=self.device, generator=generator)
            else:
                pixel_attributions = torch.rand((num_targets, num_patches), device=self.device, generator=generator)
        else:
            raise ValueError(f"pixel_values shape {pixel_values.shape} is not supported.")

        return token_attributions, pixel_attributions
