from typing import Dict, Any, Optional

import numpy as np
import shapiq
import torch

from src.explainers.fixlip_utils.fixlip import FIxLIP
from src.models import BaseVLMWrapper


class GenerativeVLMGame(shapiq.Game):
    """
    An optimized shapiq.Game implementation for modern Generative VLMs 
    (LLaVA, Qwen-VL, InternVL) utilizing GPU-accelerated Embedding-Level Masking.
    
    Bypasses the ViT and tokenizer loops during sampling by masking the 
    interleaved inputs_embeds tensor directly.
    """

    def __init__(
        self,
        wrapper: BaseVLMWrapper,
        image: Any,
        text: str,
        target_ids: torch.Tensor,
        target_seq_index: int,
        batch_size: int = 32,
        patch_group_size: int = 2,
        group_text_by_word: bool = True,
    ):
        self.wrapper = wrapper
        self.model = wrapper.model
        self.device = wrapper.device
        self.batch_size = batch_size
        self.patch_group_size = patch_group_size
        self.group_text_by_word = group_text_by_word

        # 1. Base input tracking
        with torch.no_grad():
            raw_inputs = self.wrapper.get_inputs(image, text)
            self.input_ids = raw_inputs["input_ids"]
            self.base_attention_mask = raw_inputs.get("attention_mask", None)
            
            text_embeds = self.wrapper.embed_text(self.input_ids)
            kwargs_dict = {k: v for k, v in raw_inputs.items() if k not in ["pixel_values"]}
            image_embeds = self.wrapper.embed_images(raw_inputs["pixel_values"], **kwargs_dict)
            
            self.base_embeds = self.wrapper.merge_embeddings(text_embeds, image_embeds, self.input_ids).detach()
            

            self.image_token_mask = (self.input_ids == self.wrapper.special_token_ids[0])

        self.seq_len = self.base_embeds.shape[1]
        self.hidden_dim = self.base_embeds.shape[2]
        
        self.target_ids_expanded = target_ids.unsqueeze(0)
        self.target_seq_index = target_seq_index
        self.target_token_id = int(target_ids[target_seq_index].item())

        self.mask_embedding = torch.zeros((1, 1, self.hidden_dim), device=self.device, dtype=self.base_embeds.dtype)

        # 2. Map coordinates (CRITICAL: Image Players First to align with FIxLIP layout)
        self.token_player_map = torch.full((self.seq_len,), -1, dtype=torch.long, device=self.device)
        self._build_player_mapping()

        super().__init__(
            n_players=self.n_players_image + self.n_players_text, 
            normalize=True, 
            normalization_value=0.0
        )
        
        # Pre-calculate baseline normalization value
        empty_mask = np.zeros((1, self.n_players), dtype=np.int32)
        self.normalization_value = self.value_function(empty_mask)[0]

    def _build_player_mapping(self):
        """
        Maps token sequences tracking IMAGE players first, then TEXT players second.
        Includes a dynamic heuristic to group sub-word text tokens into single word players.
        """
        mask_1d = self.image_token_mask.squeeze()
        flat_input_ids = self.input_ids.squeeze() 
        
        image_sequence_indices = torch.where(mask_1d)[0]
        text_sequence_indices = torch.where(~mask_1d)[0]

        # --- A. Map Image Players First (IDs: 0 to n_players_image - 1) ---
        total_image_tokens = len(image_sequence_indices)
        grid_h = grid_w = int(np.sqrt(total_image_tokens))
        
        self.grouped_h = max(1, grid_h // self.patch_group_size)
        self.grouped_w = max(1, grid_w // self.patch_group_size)
        self.n_players_image = self.grouped_h * self.grouped_w

        for local_token_idx, seq_idx in enumerate(image_sequence_indices):
            r = (local_token_idx // grid_w) // self.patch_group_size
            c = (local_token_idx % grid_w) // self.patch_group_size
            player_idx = (r * self.grouped_w) + c
            self.token_player_map[seq_idx] = player_idx

        # --- B. Map Text Players Second (Grouped by Word) ---
        valid_text_players = 0
        tokenizer = self.wrapper.processor.tokenizer
        
        # Flag to track if we need to increment the player ID for a new word
        is_new_word = True 

        for seq_idx in text_sequence_indices:
            token_id = flat_input_ids[seq_idx].item()
            
            # Shield special tokens
            is_special = False
            if self.wrapper.special_token_ids is not None:
                if token_id in self.wrapper.special_token_ids:
                    is_special = True
            
            if not is_special:
                if self.group_text_by_word:
                    # Decode the token to inspect its string format
                    token_str = tokenizer.convert_ids_to_tokens(token_id)
                    
                    # SentencePiece uses ' ' (U+2581), BPE uses 'Ġ' for new words.
                    # Standard strings might just start with a space.
                    if token_str.startswith(' ') or token_str.startswith('Ġ') or token_str.startswith(' '):
                        is_new_word = True
                        
                    if is_new_word:
                        valid_text_players += 1
                        is_new_word = False # Lock the ID for subsequent sub-tokens
                else:
                    # Standard 1-to-1 token mapping
                    valid_text_players += 1
                
                # Assign player index offset to sit right after image players.
                # We subtract 1 because valid_text_players acts as a 1-based counter here.
                self.token_player_map[seq_idx] = self.n_players_image + valid_text_players - 1
                
        self.n_players_text = valid_text_players

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluates flat joint coalitions matrix [batch_size, n_image + n_text]"""
        budget = coalitions.shape[0]
        game_values = np.zeros(budget, dtype=np.float32)
        coalitions_tensor = torch.tensor(coalitions, device=self.device, dtype=torch.long)

        for batch_start in range(0, budget, self.batch_size):
            current_bs = min(self.batch_size, budget - batch_start)
            batch_end = batch_start + current_bs
            
            batch_coalitions = coalitions_tensor[batch_start:batch_end]
            batch_embeds = self._apply_embedding_mask(batch_coalitions, current_bs)
            
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=batch_embeds,
                    attention_mask=self.base_attention_mask.expand(current_bs, -1) if self.base_attention_mask is not None else None
                )
                target_logits = outputs.logits[:, -1, :] # Slices tracking sequence predicting target_seq_index
                log_probs = torch.log_softmax(target_logits, dim=-1)
                probs = torch.exp(log_probs[:, self.target_token_id])
                
            game_values[batch_start:batch_end] = probs.float().cpu().numpy()
            
        return game_values

    def value_function_crossmodal(self, coalitions_image: np.ndarray,
                                  coalitions_text: np.ndarray) -> np.ndarray:
        """
        Natively handles 2D tiled layout optimization required by your FIxLIP script,
        processing evaluations efficiently via GPU batch broadcasting blocks.
        """
        n_img_masks = coalitions_image.shape[0]
        n_txt_masks = coalitions_text.shape[0]
        
        # Replicate the cross-modal tile layouts to form a unified flat execution space
        combined_p_mask = np.repeat(coalitions_image, n_txt_masks, axis=0)
        combined_t_mask = np.tile(coalitions_text, (n_img_masks, 1))
        
        # Concat along features dimension matching our Image-First sequence configuration
        flat_coalitions = np.concatenate([combined_p_mask, combined_t_mask], axis=1)
        
        # Forward through our optimized batch execution pipeline
        flat_values = self.value_function(flat_coalitions)
        
        # Reshape back to the 2D layout grid expected by FIxLIP code
        return flat_values.reshape(n_img_masks, n_txt_masks)

    def _apply_embedding_mask(self, batch_coalitions: torch.Tensor,
                              current_bs: int) -> torch.Tensor:
        batch_embeds = self.base_embeds.expand(current_bs, -1, -1).clone()
        expanded_map = self.token_player_map.expand(current_bs, -1)
        
        mask_condition = torch.gather(
            torch.cat([batch_coalitions, torch.ones((current_bs, 1), device=self.device, dtype=torch.long)], dim=1),
            dim=1,
            index=torch.where(expanded_map < 0, batch_coalitions.shape[1], expanded_map)
        )
        
        absent_indices = (mask_condition == 0).unsqueeze(-1)
        batch_embeds = torch.where(absent_indices, self.mask_embedding, batch_embeds)
        return batch_embeds
    
