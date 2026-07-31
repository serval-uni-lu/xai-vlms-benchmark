from typing import Any, Optional
import copy

import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from src.models import BaseVLMWrapper
from src.explainers import BaseExplainer
from .eagle_utils.text_attribution import UniversalEAGLETextAdaptor, EfficientLLMSubModularExplanationText
from .eagle_utils.pixel_attribution import UniversalEAGLEVisionAdaptor, EfficientMLLMSubModularExplanationVision
from .eagle_utils.utils import build_text_span_masks_from_inputs, SubRegionDivision

class EagleExplainer(BaseExplainer):
    """
    A unified explainer that extracts both visual sub-region attributions 
    and text token attributions using EAGLE's greedy submodular optimization,
    fully routed through your agnostic BaseVLMWrapper.
    """

    def __init__(self, model_wrapper: BaseVLMWrapper,
                division_number: int = 50,
                text_kwargs: Optional[dict[str, Any]] = None,
                vision_kwargs: Optional[dict[str, Any]] = None,
                ):
        """
        Args:
            model_wrapper (BaseVLMWrapper): Your custom unified VLM wrapper.
            division_number (int): Number of sub-regions to segment the image into for vision attribution.
            eagle_kwargs: Keyword arguments passed directly to the EAGLE explanation submodules.
        """
        super().__init__(model_wrapper)
        self.division_number = division_number
        self.text_kwargs: dict[str, Any] = text_kwargs if text_kwargs is not None else {}
        self.vision_kwargs: dict[str, Any] = vision_kwargs if vision_kwargs is not None else {}
        
        self.vision_adaptor = UniversalEAGLEVisionAdaptor(copy.deepcopy(self.wrapper))
        self.text_adaptor = UniversalEAGLETextAdaptor(copy.deepcopy(self.wrapper))

    def _attribute(
        self, image: Image.Image | torch.Tensor, text: str,
        target_indices: int | list[int] | None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes text and image attributions for one or multiple target tokens.
        Optimized to calculate image regions and text masks only once.
        """
        # 1. Standardize targets
        if isinstance(target_indices, int):
            target_indices = [target_indices]
        target_indices_np = np.array(target_indices)

        if isinstance(image, torch.Tensor):
            image_pil = transforms.ToPILImage()(image.cpu())
        else:
            image_pil = image

        # 2. Extract base inputs and generated sequence
        base_inputs = self.wrapper.get_inputs(image=image_pil, text=text)
        prompt_ids = base_inputs["input_ids"][0].to(self.device)
        prompt_len = len(prompt_ids)

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(
                base_inputs,
                return_logits=True,
            )
        generated_answer_ids = pred_results["new_ids"].to(self.device)
        
        max_requested_idx = target_indices_np.max()
        if max_requested_idx >= len(generated_answer_ids):
            raise IndexError(
                f"Target index {max_requested_idx} is out of bounds. "
                f"The model only generated {len(generated_answer_ids)} new tokens."
            )

        generated_ids = torch.cat([prompt_ids, generated_answer_ids], dim=0).unsqueeze(0)

        # ---------------------------------------------------------------------
        # PRE-LOOP OPTIMIZATION: Build Regions and Masks Once
        # ---------------------------------------------------------------------
        # Image Regions
        raw_cv_img = np.array(image_pil)
        region_size = int((raw_cv_img.shape[0] * raw_cv_img.shape[1] / self.division_number) ** 0.5)
        vision_v_set = SubRegionDivision(raw_cv_img, region_size=region_size)

        # Text Spans
        input_ids = base_inputs["input_ids"][0]
        text_v_set, spans = build_text_span_masks_from_inputs(
            processor=self.wrapper.processor,
            input_ids=input_ids,
            target_text=text,
        )

        # Initialize storage for batch attributions
        all_text_attrs = []
        all_img_attrs = []

        # ---------------------------------------------------------------------
        # MAIN LOOP: Process Each Target Token Independently
        # ---------------------------------------------------------------------
        for target_idx in target_indices_np:
            absolute_pos = int(target_idx + prompt_len)
            word_id = int(generated_ids[0, absolute_pos].item())
            # word_id = generated_ids[0, absolute_pos].cpu().tolist()

            # --- A. Vision Attribution Pass ---
            self.vision_adaptor.generated_ids = generated_ids
            self.vision_adaptor.text_prompt = text
            self.vision_adaptor.target_token_position = np.array([absolute_pos])
            self.vision_adaptor.selected_interpretation_token_word_id = [word_id]

            self.vision_explainer = EfficientMLLMSubModularExplanationVision(
                self.vision_adaptor, **self.vision_kwargs
            )
            
            visual_s_set, saved_visual_json_file = self.vision_explainer(
                image=raw_cv_img, V_set=vision_v_set
            )
            
            img_attrs_single = self._compute_pixel_relevance(
                visual_s_set, saved_visual_json_file, raw_cv_img
            )
            img_attrs_single = self._format_pixel_attributions(img_attrs_single, base_inputs)
            all_img_attrs.append(img_attrs_single)

            # --- B. Text Attribution Pass ---
            self.text_adaptor.prompt_length = prompt_len
            self.text_adaptor.generated_answer_ids = generated_answer_ids
            self.text_adaptor.target_token_position = torch.tensor([absolute_pos], device=self.device)
            self.text_adaptor.selected_interpretation_token_word_id = torch.tensor([word_id], device=self.device)
            
            self.text_adaptor.static_vision_inputs = {
                k: v for k, v in base_inputs.items() if k not in ("input_ids", "attention_mask")
            }

            self.text_explainer = EfficientLLMSubModularExplanationText(
                model=self.text_adaptor, **self.text_kwargs
            )
            
            token_s_set, _ = self.text_explainer(
                source_input_ids=input_ids, V_set=text_v_set
            )
            text_attrs_single = self._compute_token_relevance(token_s_set, input_ids, spans)
            all_text_attrs.append(text_attrs_single)

        # ---------------------------------------------------------------------
        # POST-PROCESSING: Stack and Shape
        # ---------------------------------------------------------------------
        # Stack into shapes: (Num_Targets, Seq_Len) and (Num_Targets, Num_Patches)
        final_text_attrs = torch.stack(all_text_attrs, dim=0)
        final_img_attrs = torch.stack(all_img_attrs, dim=0)

        # If only one token was requested, drop the batch dimension to maintain API consistency
        # if len(target_indices_np) == 1:
        #     final_text_attrs = final_text_attrs.squeeze(0)
        #     final_img_attrs = final_img_attrs.squeeze(0)

        return final_text_attrs, final_img_attrs

    def _compute_pixel_relevance(self, S_set, saved_json_file, raw_cv_img):
        # 2. Calculate the Submodular Score (SMDL) if using V2
        # EAGLE combines Insertion and Deletion scores to get the final confidence metric
        if "insertion_score" in saved_json_file and "deletion_score" in saved_json_file:
            saved_json_file["smdl_score"] = (
                np.array(saved_json_file["insertion_score"]) + 1 - np.array(saved_json_file["deletion_score"])
            ).tolist()

        # 3. Build the Spatial Attribution Map
        if "smdl_score" in saved_json_file:
            # This function paints the smdl_scores back onto the irregular SLIC sub-regions
            attribution_map, _ = self._add_value(S_set, saved_json_file)
            
            # attribution_map is usually returned as (H, W, Channels). Extract a single 2D grid.
            img_attrs_np = attribution_map[:, :, 0] if attribution_map.ndim == 3 else attribution_map

        else:
            # Fallback for V1: S_set itself is the raw mask grid.
            attribution_map = np.array(S_set, dtype=np.float32)
            
            # Min-Max Normalization
            attribution_map = attribution_map - attribution_map.min()
            attribution_map = attribution_map / (attribution_map.max() + 1e-8)
            
            # Upsample the coarse grid to the original image dimensions using Bilinear interpolation
            img_attrs_np = cv2.resize(
                attribution_map, 
                (raw_cv_img.shape[1], raw_cv_img.shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )

        # 4. Convert to PyTorch Tensor for your XAIResult container
        img_attrs = torch.tensor(img_attrs_np, dtype=torch.float32)
        return img_attrs
    
    def _add_value(self, S_set, json_file):
        single_mask = np.zeros_like(S_set[0])
        single_mask = single_mask.astype(np.float16)
        
        value_list_1 = np.array(json_file["smdl_score"])
        
        # value_list_2 = np.array(
        #     [1 - np.mean(json_file["org_score"]) + np.mean(json_file["baseline_score"])] + json_file["smdl_score"][:-1]
        # )
        value_list_2 = np.array(
            [np.mean(1 - np.array(json_file["org_score"]) + np.array(json_file["baseline_score"]))] + json_file["smdl_score"][:-1]
        )
        
        # value_list = np.exp((value_list_1 - value_list_2)/1)
        value_list = value_list_1 - value_list_2
        
        values = []
        value = 0
        i = 0
        for smdl_single_mask, smdl_value in zip(S_set, value_list):
            value = value - abs(smdl_value)
            single_mask[smdl_single_mask==1] = value
            values.append(value)
            i+=1
        attribution_map = single_mask - single_mask.min()
        attribution_map = attribution_map / attribution_map.max()
        
        return attribution_map, np.array(values)

    def _compute_token_relevance(self, S_set, input_ids, spans):
        # 1. Initialize a zero tensor for the full sequence length
        seq_len = input_ids.shape[-1] if input_ids.dim() > 1 else len(input_ids)
        text_attrs = torch.zeros(seq_len, dtype=torch.float32)

        # 2. Safely extract the exact order of span selection, handling nested/cumulative arrays
        flat_ordered_indices = []
        seen_indices = set()

        for step_result in S_set:
            # Normalize PyTorch tensors to NumPy arrays
            if isinstance(step_result, torch.Tensor):
                step_result = step_result.detach().cpu().numpy()
            
            # atleast_1d and flatten handles single scalars, lists, and nested arrays
            items = np.atleast_1d(step_result).flatten()
            
            for idx in items:
                clean_idx = int(idx)
                # Only record the span the *first* time it is selected
                if clean_idx not in seen_indices:
                    seen_indices.add(clean_idx)
                    flat_ordered_indices.append(clean_idx)

        # 3. Convert rankings into numerical scores
        num_spans = len(spans)

        for rank_idx, span_index in enumerate(flat_ordered_indices):
            importance_score = float(num_spans - rank_idx)
            target_span = spans[span_index]
            
            # Scatter the importance score to all tokens that belong to this word/span
            for token_idx in target_span.token_indices:
                text_attrs[token_idx] = importance_score

        # Optional: Normalize the final tensor to be between 0 and 1 for your benchmark
        # if text_attrs.max() > 0:
        #     text_attrs = text_attrs / text_attrs.max()
        
        return text_attrs

    def _format_pixel_attributions(
            self,
            img_attrs_orig: torch.Tensor, 
            inputs: dict[str, Any],
        ) -> torch.Tensor:
        """
        Casts a (H_orig, W_orig) heatmap into the model's processed visual topology.
        """

        model_type = self.wrapper.model.config.model_type
        pixel_values = inputs["pixel_values"]

        # 1. Add batch and channel dimensions for PyTorch interpolation -> (1, 1, H, W)
        if img_attrs_orig.dim() == 2:
            heatmap_4d = img_attrs_orig.unsqueeze(0).unsqueeze(0)
        else:
            heatmap_4d = img_attrs_orig

        # ==========================================
        # CASE A: Standard VLM (e.g., LLaVA)
        # pixel_values shape: (Batch, C, H_model, W_model)
        # Target attribution shape: (H_model, W_model)
        # ==========================================
        if "llava" in model_type:
            # Extract the target H and W directly from the pixel_values tensor
            h_model, w_model = pixel_values.shape[-2], pixel_values.shape[-1]
            
            # Downsample the attribution map to the ViT resolution (e.g., 336x336)
            resized = F.interpolate(
                heatmap_4d, 
                size=(h_model, w_model), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Strip batch and channel -> Shape: (H_model, W_model)
            return resized.squeeze()

        # ==========================================
        # CASE B: Packed Patches VLM (e.g., Qwen-VL)
        # pixel_values shape: (Batch, num_patches, patch_dim)
        # Target attribution shape: (num_patches,)
        # ==========================================
        elif "qwen" in model_type:
            # Qwen provides the 2D grid dimensions in the kwargs
            if "image_grid_thw" not in inputs:
                raise ValueError("Qwen model requires 'image_grid_thw' to map patches.")
                
            # Extract the patch grid height and width (Temporal is usually index 0)
            _, grid_h, grid_w  = inputs["image_grid_thw"][0].cpu().numpy().tolist()
            
            # Resize the heatmap so every pixel corresponds to exactly one model patch
            resized = F.interpolate(
                heatmap_4d, 
                size=(grid_h, grid_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Flatten the 2D grid into the 1D sequence the model expects -> Shape: (num_patches,)
            return resized.flatten()

        # ==========================================
        # CASE C: AnyRes Tiling (e.g., InternVL)
        # pixel_values shape: (Batch, num_tiles, C, H_tile, W_tile)
        # Target attribution shape: (num_tiles, H_tile, W_tile)
        # ==========================================
        elif "internvl" in model_type:
            # Extract tile counts and dimensions
            num_tiles, C, h_tile, w_tile = pixel_values.shape
            # num_tiles = pixel_values.shape[1]
            # h_tile, w_tile = pixel_values.shape[-2], pixel_values.shape[-1]
            
            # *Benchmark Note*: InternVL dynamically slices the original image into local crops 
            # plus one global downsampled thumbnail. To make the heatmap perfectly match the 
            # pixels of each tile, the most rigorous method is to pass the heatmap tensor 
            # through the exact same torchvision transforms (crop/pad) used by the processor.
            
            # Fallback/Proxy implementation: Resize the heatmap to the tile resolution
            # and duplicate it across the tile dimension.
            resized_tile = F.interpolate(
                heatmap_4d, 
                size=(h_tile, w_tile), 
                mode='bilinear', 
                align_corners=False
            ).squeeze() # Shape: (H_tile, W_tile)
            
            # Stack it to match the number of tiles -> Shape: (num_tiles, H_tile, W_tile)
            tiled_heatmap = resized_tile.unsqueeze(0).expand(num_tiles, -1, -1)
            
            return tiled_heatmap

        else:
            raise ValueError(f"Unknown VLM mapping format: {model_type}")
