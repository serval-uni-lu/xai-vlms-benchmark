import torch
import torch.nn.functional as F

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper


class OracleExplainer(BaseExplainer):
    """
    An Oracle (Ground Truth) explainer that returns perfect attributions.
    Used to validate the upper bound of faithfulness metrics (Phase 1.1).
    Compatible with Standard 4D, QwenVL 3D (patches), and InternVL 5D (tiles).
    """

    def __init__(self, model_wrapper: "BaseVLMWrapper"):
        super().__init__(model_wrapper)

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates ground-truth attention maps for Image and Text.
        Requires 'keyword' and 'oracle_mask_2d' passed via kwargs.
        """

        # --- EXTRACT ORACLE KWARGS ---
        keyword = kwargs.get("keyword")
        oracle_mask_2d = kwargs.get("oracle_mask_2d")  # Expected shape: (H_img, W_img)

        if keyword is None or oracle_mask_2d is None:
            raise ValueError("OracleExplainer requires 'keyword' and 'oracle_mask_2d' in kwargs.")

        # 1. Prepare Inputs using the wrapper's processor
        inputs = self.wrapper.get_inputs(image, text)
        input_ids = inputs["input_ids"]  # Shape: (1, seq_len)
        pixel_values = inputs["pixel_values"]

        # --- SAFELY ADD BATCH DIMENSION ---
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

        # Retrieve predictions to calculate targets
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

        # ==========================================
        # 2. BUILD TOKEN ORACLE
        # ==========================================
        base_token_mask = self._build_token_mask(input_ids[0], text, keyword)
        # Expand the single 1D mask to match the number of target generated tokens
        token_attributions = base_token_mask.unsqueeze(0).expand(num_targets, -1)

        # ==========================================
        # 3. BUILD PIXEL ORACLE
        # ==========================================
        base_pixel_mask = self._build_pixel_mask(pixel_values, oracle_mask_2d, model_type, inputs)
        # Expand to match num_targets
        pixel_attributions = base_pixel_mask.unsqueeze(0).expand(num_targets, *base_pixel_mask.shape)

        return token_attributions, pixel_attributions

    def _build_token_mask(self, true_input_ids: torch.Tensor, prompt: str, keyword: str) -> torch.Tensor:
        """Finds the keyword in the prompt and matches it to the true input_ids tensor."""
        tokenizer = self.wrapper.processor.tokenizer
        text_only_ids = tokenizer.encode(prompt, add_special_tokens=False)

        keyword_token_ids = None
        clean_keyword = keyword.lower().strip()
        seq_len = len(text_only_ids)

        # Sliding window to find how the BPE tokenizer split the keyword
        for i in range(seq_len):
            for j in range(i + 1, min(i + 6, seq_len + 1)):
                decoded = tokenizer.decode(text_only_ids[i:j]).lower().strip()
                if decoded == clean_keyword:
                    keyword_token_ids = text_only_ids[i:j]
                    break
            if keyword_token_ids is not None:
                break

        token_mask = torch.zeros_like(true_input_ids, dtype=torch.float32)

        # Apply the 1.0 mask to the exact position in the true input tensor
        if keyword_token_ids is not None:
            target_seq = torch.tensor(keyword_token_ids, device=true_input_ids.device)
            target_len = len(target_seq)
            for i in range(len(true_input_ids) - target_len + 1):
                if torch.equal(true_input_ids[i : i + target_len], target_seq):
                    token_mask[i : i + target_len] = 1.0
                    break
        else:
            print(f"Warning: Keyword '{keyword}' not found by Oracle Token Builder.")

        return token_mask

    def _build_pixel_mask(
        self, pixel_values: torch.Tensor, mask_2d: torch.Tensor, model_type: str, inputs: dict
    ) -> torch.Tensor:
        """Reshapes the raw 2D COCO mask into the specific architecture format."""
        # Prep mask for interpolation (1, 1, H, W)
        mask = mask_2d.unsqueeze(0).unsqueeze(0).float().to(self.device)

        if "internvl" in model_type:
            # (B, num_tiles, C, H, W) -> Returns (num_tiles, H, W)
            _, num_tiles, _, h_p, w_p = pixel_values.shape
            # Approximate the tiling by resizing the mask to the tile size and duplicating
            resized = F.interpolate(mask, size=(h_p, w_p), mode="nearest").squeeze()
            return resized.unsqueeze(0).expand(num_tiles, -1, -1)

        elif "llava" in model_type:
            # (B, C, H, W) -> Returns (H, W)
            _, _, h_p, w_p = pixel_values.shape
            resized = F.interpolate(mask, size=(h_p, w_p), mode="nearest").squeeze()
            return resized

        elif "qwen" in model_type:
            # (B, num_patches, C) -> Returns (H * W)
            if "image_grid_thw" in inputs:
                grid_thw = inputs["image_grid_thw"][0].cpu().numpy().tolist()
                h_p, w_p = grid_thw[1], grid_thw[2]
                resized = F.interpolate(mask, size=(h_p, w_p), mode="nearest").squeeze()
                return resized.flatten()
            else:
                # Fallback for Qwen if grid info is missing
                _, num_patches, _ = pixel_values.shape
                dim = int(num_patches**0.5)
                resized = F.interpolate(mask, size=(dim, dim), mode="nearest").squeeze()
                flat = resized.flatten()
                # Pad/truncate to exact num_patches just in case
                if len(flat) > num_patches:
                    flat = flat[:num_patches]
                elif len(flat) < num_patches:
                    flat = F.pad(flat, (0, num_patches - len(flat)))
                return flat

        else:
            raise ValueError(f"Oracle pixel reshaping not supported for {model_type}")


class AntiExplainer(OracleExplainer):
    """
    The Anti-Explainer. Generates maliciously wrong attributions by
    perfectly inverting the Oracle's ground-truth masks.
    Used to validate the lower bound of faithfulness metrics.
    """

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 1. Let the Oracle do all the hard shape-matching work!
        perfect_tokens, perfect_pixels = super()._attribute(
            image=image, text=text, target_indices=target_indices, **kwargs
        )

        # 2. INVERT THE MASKS
        # The object/keyword (1.0) becomes hidden (0.0).
        # The background/grammar (0.0) becomes maximally important (1.0).
        anti_tokens = 1.0 - perfect_tokens
        anti_pixels = 1.0 - perfect_pixels

        return anti_tokens, anti_pixels


class MismatchedExplainer(OracleExplainer):
    """
    The Deceptive Explainer (For Experiment 1.2).
    Generates perfect text attributions, but uses a mismatched/wrong image mask.
    Proves that the Synergy metric catches cross-modal hallucinations.
    """

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # We need the CORRECT keyword to build the perfect text mask
        keyword = kwargs.get("keyword")

        # But we pass in the DECEPTIVE/WRONG mask for the image
        deceptive_mask_2d = kwargs.get("mismatched_mask_2d")

        if deceptive_mask_2d is None:
            raise ValueError("MismatchedExplainer requires 'mismatched_mask_2d'.")

        # Let the Oracle do the shape-matching work, but feed it the lie!
        perfect_tokens, wrong_pixels = super()._attribute(
            image=image,
            text=text,
            target_indices=target_indices,
            keyword=keyword,
            oracle_mask_2d=deceptive_mask_2d,  # <-- THE BAIT AND SWITCH
            pred_results=kwargs.get("pred_results"),
        )

        return perfect_tokens, wrong_pixels
