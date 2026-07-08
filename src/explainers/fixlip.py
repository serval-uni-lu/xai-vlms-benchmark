from typing import Dict, Any, Optional
import time

import numpy as np
import shapiq
import torch

from src.models import BaseVLMWrapper
from src.explainers import BaseExplainer
from src.explainers.fixlip_utils.fixlip import FIxLIP
from src.explainers.fixlip_utils.utils import convert_iv_to_first_order
from src.explainers.fixlip_utils.vlm_game import GenerativeVLMGame
from src.utils.xai_utils import align_llm_visuals_to_pixels



class FIxLIPExplainer(BaseExplainer):
    """
    Computes cross-modal feature attributions using ProxySHAP and the weighted 
    Banzhaf power index (or Shapley) via the FIxLIP algorithm.
    """

    def __init__(
        self, 
        model_wrapper: BaseVLMWrapper, 
        budget: int = 1024, # Total cross-modal budget
        approximation_type: str = "proxyshap",
        mode: str = "banzhaf",
        adjustment: str = "msr",
        max_order: int = 2,
        batch_size: int = 32,
        mask_value: float = 0.0,
        patch_group_size: int = 2,
        p_weight: float = 0.5,
        group_text_by_word: bool = True,
        seed: int = 42,
        proxy_xgboost_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_wrapper)
        self.budget = budget
        self.batch_size = batch_size
        self.approximation_type = approximation_type
        self.mode = mode
        self.adjustment = adjustment
        self.max_order = max_order
        self.mask_value = mask_value
        self.p_weight = p_weight
        self.patch_group_size = patch_group_size
        self.group_text_by_word = group_text_by_word
        self.seed = seed
        self.proxy_xgboost_kwargs: dict[str, Any] = proxy_xgboost_kwargs \
            if proxy_xgboost_kwargs is not None else {}

    def _attribute(
        self,
        image,
        text: str,
        target_indices: Optional[int | list[int]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Prepare Inputs using the wrapper's processor
        inputs = self.wrapper.get_inputs(image, text)
        pixel_values_shape = inputs["pixel_values"].shape

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

        # 2. Storage for final stacked tensors
        all_token_attrs = []
        all_pixel_attrs = []

        # 3. Game-Theoretic Evaluation Loop
        for target_seq_idx in indices_to_compute:
            # Initialize the Game for this specific target token
            vlm_game = GenerativeVLMGame(
                wrapper=self.wrapper,
                image=image,
                text=text,
                target_ids=new_ids,
                target_seq_index=target_seq_idx,
                batch_size=self.batch_size,
                patch_group_size=self.patch_group_size,
                group_text_by_word=self.group_text_by_word,
            )

            # Initialize FIxLIP Explainer
            fixlip = FIxLIP(
                n_players_image=vlm_game.n_players_image,
                n_players_text=vlm_game.n_players_text,
                max_order=self.max_order,
                p=self.p_weight,
                mode=self.mode,
                random_state=self.seed,
            )

            # Compute interaction matrix
            start_time = time.time()
            self.interaction_values = fixlip.approximate_crossmodal(
                game=vlm_game,
                budget=self.budget,
                approximation_type=self.approximation_type,
                **self.proxy_xgboost_kwargs
            )
            print(f"[FIxLIP] Computed token index {target_seq_idx} in {time.time() - start_time:.2f}s")

            # Collapse 2nd-order synergies to 1st-order main effects
            attribution_values = convert_iv_to_first_order(self.interaction_values)

            # Extract the raw 1D tensors
            text_scores, flat_image_attr = self._extract_feature_attributions(
                game=vlm_game, 
                first_order_iv=attribution_values
            )

            # Move to device and store
            text_tensor = text_scores.to(self.device).squeeze()
            image_tensor = flat_image_attr.to(self.device)

            all_token_attrs.append(text_tensor)
            all_pixel_attrs.append(image_tensor)

        # 4. Stack arrays
        # Token attributions shape: [gen_len, prompt_len]
        token_attributions = torch.stack(all_token_attrs)
        
        # Raw pixel attributions shape: [gen_len, num_llm_tokens]
        raw_pixel_attributions = torch.stack(all_pixel_attrs)

        # 5. Apply your external alignment function
        # This will reshape and interpolate the entire batch natively based on model_type
        aligned_pixel_attributions = align_llm_visuals_to_pixels(
            pixel_attribution=raw_pixel_attributions,
            inputs=inputs,
            config=self.wrapper.model.config
        )

        return token_attributions, aligned_pixel_attributions
    
    def _extract_feature_attributions(self, game: GenerativeVLMGame,
                                      first_order_iv: shapiq.InteractionValues
                                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes player IDs back to a text sequence matching input_ids, 
        and a flat 1D array of image attributions matching the LLM visual tokens.
        """
        player_map = game.token_player_map.cpu()
        
        # ---------------------------------------------------------
        # 1. EXTRACT TEXT ATTRIBUTIONS
        # ---------------------------------------------------------
        text_attributions_seq = torch.zeros(game.input_ids.shape, dtype=torch.float32)
        
        for player_tuple, score in first_order_iv.dict_values.items():
            if len(player_tuple) == 0:
                continue
                
            player_id = player_tuple[0]
            
            if player_id >= game.n_players_image:
                seq_indices = torch.where(player_map == player_id)[0]
                for seq_idx in seq_indices:
                    seq_idx = int(seq_idx.item())
                    if text_attributions_seq.ndim == 2:
                        text_attributions_seq[0, seq_idx] = score
                    else:
                        text_attributions_seq[seq_idx] = score

        # ---------------------------------------------------------
        # 2. EXTRACT & UN-GROUP RAW IMAGE LLM TOKENS
        # ---------------------------------------------------------
        image_seq_indices = torch.where(game.image_token_mask.squeeze())[0]
        num_llm_tokens = len(image_seq_indices)
        
        # Create the flat array expected by align_llm_visuals_to_pixels [num_llm_tokens]
        flat_image_attr = torch.zeros(num_llm_tokens, dtype=torch.float32)
        
        for local_idx, seq_idx in enumerate(image_seq_indices):
            player_id = int(game.token_player_map[seq_idx].item())
            # Fetch the score; defaults to 0.0 if that patch group wasn't sampled
            flat_image_attr[local_idx] = first_order_iv.dict_values.get((player_id,), 0.0)

        return text_attributions_seq, flat_image_attr

