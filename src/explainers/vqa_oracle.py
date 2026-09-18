import numpy as np
import torch

from src.explainers.base import BaseExplainer


class GridVQAOracleExplainer(BaseExplainer):
    """
    An Oracle Explainer specifically designed for the GridVQA dataset.
    - Uses pre-model spatial masks for the image.
    - Dynamically generates token masks to match the VLM's exact chat-templated input_ids.
    """

    def __init__(self, model_wrapper):
        super().__init__(model_wrapper)

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # --- 1. EXTRACT GRIDVQA METADATA ---
        grid_metadata = kwargs.get("grid_metadata")
        if grid_metadata is None:
            raise ValueError("GridVQAOracleExplainer requires 'grid_metadata' passed via kwargs.")

        # 2. Prepare Inputs using the wrapper (THIS APPLIES CHAT TEMPLATES!)
        inputs = self.wrapper.get_inputs(image, text)
        input_ids = inputs["input_ids"]  # Shape: (1, final_seq_len)

        # Retrieve predictions to calculate target length
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

        if target_indices is None:
            indices_to_compute = list(range(seq_len_generated))
        elif isinstance(target_indices, int):
            indices_to_compute = [target_indices]
        else:
            indices_to_compute = target_indices

        indices_to_compute = [idx for idx in indices_to_compute if idx < seq_len_generated]
        num_targets = max(1, len(indices_to_compute))

        # ==========================================
        # 3. BUILD TRUE TOKEN ORACLE (Chat-Template Safe)
        # ==========================================
        # Notice we use "raw_objects" from our updated dataset metadata
        objects_list = grid_metadata.get("raw_objects", grid_metadata.get("objects", []))
        
        # We pass input_ids[0] so the mask is built EXACTLY to the final sequence length
        base_token_mask = self._build_token_mask_from_objects(input_ids[0], objects_list)
        token_attributions = base_token_mask.unsqueeze(0).expand(num_targets, -1)

        # ==========================================
        # 4. BUILD TRUE PIXEL ORACLE (Grid-Safe)
        # ==========================================
        # We directly use the perfectly aligned (H, W) mask from the Dataset!
        if "model_pixel_mask" in grid_metadata:
            base_pixel_mask = grid_metadata["model_pixel_mask"].to(self.device).float()
        else:
            raise KeyError("Missing 'model_pixel_mask' in metadata. Ensure the Dataset provides it.")
            
        pixel_attributions = base_pixel_mask.unsqueeze(0).expand(num_targets, *base_pixel_mask.shape)

        return token_attributions, pixel_attributions

    def _build_token_mask_from_objects(
        self, 
        input_ids_1d: torch.Tensor, 
        objects: list | np.ndarray
    ) -> torch.Tensor:
        """
        Dynamically finds the tokens corresponding to the GridVQA objects 
        within the fully tokenized prompt (including chat templates).
        """
        tokenizer = self.wrapper.processor.tokenizer
        device = input_ids_1d.device
        mask = torch.zeros_like(input_ids_1d, dtype=torch.float32, device=device)
        
        ids_list = input_ids_1d.tolist()
        
        # Reconstruct the string from the exact tokens to capture special tokens/templates
        current_str = ""
        token_char_spans = []
        
        for tid in ids_list:
            token_str = tokenizer.decode([tid], skip_special_tokens=False)
            start_char = len(current_str)
            current_str += token_str
            end_char = len(current_str)
            token_char_spans.append((start_char, end_char, token_str))
            
        prompt_lower = current_str.lower()
        search_terms = set()
        
        if isinstance(objects, np.ndarray):
            objects = objects.tolist()
            
        for obj in objects:
            if isinstance(obj, dict):
                if 'name' in obj:
                    search_terms.add(str(obj['name']).lower().strip())
                else:
                    color = str(obj.get('color', '')).lower().strip()
                    shape = str(obj.get('shape', '')).lower().strip()
                    if color and shape:
                        search_terms.add(f"{color} {shape}")
                    if color:
                        search_terms.add(color)
                    if shape:
                        search_terms.add(shape)
            else:
                search_terms.add(str(obj).lower().strip())
                
        for term in search_terms:
            if not term:
                continue
                
            start_idx = 0
            while True:
                target_start = prompt_lower.find(term, start_idx)
                if target_start == -1:
                    break
                target_end = target_start + len(term)
                
                for i, (start, end, text) in enumerate(token_char_spans):
                    if max(start, target_start) < min(end, target_end) and text.strip():
                        mask[i] = 1.0
                            
                start_idx = target_start + 1 
                        
        return mask

class GridVQAAntiOracleExplainer(GridVQAOracleExplainer):
    """
    Inverts the Ground Truth to act as the worst possible explainer.
    It assigns 1.0 to background/irrelevant features and 0.0 to the true causal features.
    """
    def _attribute(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Get the perfect Ground Truth from the Oracle
        token_attributions, pixel_attributions = super()._attribute(*args, **kwargs)
        
        # 2. Invert the probabilities (1.0 becomes 0.0, 0.0 becomes 1.0)
        anti_token_attr = 1.0 - token_attributions
        anti_pixel_attr = 1.0 - pixel_attributions
        
        return anti_token_attr, anti_pixel_attr