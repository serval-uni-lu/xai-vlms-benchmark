from typing import Any

import numpy as np
import shapiq
import torch
from shapiq import Game
from torch import Tensor

from src.models import BaseVLMWrapper
from src.utils.faithfulness_utils import (
    _reshape_pixels_faithfulness,
    get_most_important_tokens_multimodal,
    make_blur_baseline,
    score_output,
)


class MacroSynergyGame(Game):
    """
    A shapiq Game interface that computes model probabilities for
    specific coalitions of Top-K features and Background chunks.
    """

    def __init__(
        self,
        model: BaseVLMWrapper,
        inputs: dict[str, Any],
        target_ids: Tensor,
        target_positions,
        feat: Tensor,
        feat_baseline: Tensor,
        input_ids: Tensor,
        baseline_input_ids: Tensor,
        top_img_idx,
        top_token_idx,
        bg_img_groups,
        bg_txt_groups,
        num_img_feat,
        ndim: int,
        origin_shape,
        batch_size: int = 32,
    ):

        # Player 0: Top-K Image, Player 1: Top-K Text, Players 2..N: Background
        self.n_background = len(bg_img_groups)
        self.n_players = 2 + self.n_background

        # Initialize shapiq.Game
        super().__init__(n_players=self.n_players, normalize=False)

        self.model = model
        self.inputs = inputs
        self.target_ids = target_ids
        self.target_positions = target_positions

        # Storing the pre-flattened baselines and originals
        self.feat = feat
        self.feat_baseline = feat_baseline
        self.input_ids = input_ids
        self.baseline_input_ids = baseline_input_ids

        # Storing the player indices
        self.top_img_idx = top_img_idx
        self.top_token_idx = top_token_idx
        self.bg_img_groups = bg_img_groups
        self.bg_txt_groups = bg_txt_groups

        self.n_bg_img = len(bg_img_groups)
        self.n_bg_txt = len(bg_txt_groups)

        # Shape info
        self.num_img_feat = num_img_feat
        self.ndim = ndim
        self.origin_shape = origin_shape
        self.batch_size = batch_size

        self.model_type = getattr(model.model.config, "model_type", "").lower()

    def value_function_crossmodal(self, coalitions: np.ndarray) -> np.ndarray:
        n_coalitions = coalitions.shape[0]
        coalitions_outputs = np.zeros(n_coalitions)

        for start_idx in range(0, n_coalitions, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_coalitions)
            current_batch_size = end_idx - start_idx
            batch_coalitions = coalitions[start_idx:end_idx]

            batch_feats = self.feat_baseline.expand(current_batch_size, -1, -1).clone()
            batch_input_ids = self.baseline_input_ids.expand(current_batch_size, -1).clone()

            for i in range(current_batch_size):
                coalition = batch_coalitions[i]

                # --- A. Image Scatter (Player 0 + Image BG Players) ---
                active_img_idx_list = []
                if coalition[0] and self.top_img_idx.numel() > 0:
                    active_img_idx_list.append(self.top_img_idx.squeeze(0))  # Player 0

                # Image BG Players start at index 2
                for j in range(self.n_bg_img):
                    if coalition[2 + j] and len(self.bg_img_groups[j]) > 0:
                        active_img_idx_list.append(self.bg_img_groups[j])

                if len(active_img_idx_list) > 0:
                    active_img_idx = torch.cat(active_img_idx_list).unsqueeze(0)
                    active_img_idx_exp = active_img_idx.unsqueeze(1).expand(1, self.num_img_feat, -1)
                    pixels_orig = self.feat.gather(dim=2, index=active_img_idx_exp)
                    batch_feats[i : i + 1].scatter_(dim=2, index=active_img_idx_exp, src=pixels_orig)

                # --- B. Text Scatter (Player 1 + Text BG Players) ---
                active_txt_idx_list = []
                if coalition[1] and self.top_token_idx.numel() > 0:
                    active_txt_idx_list.append(self.top_token_idx.squeeze(0))  # Player 1

                # Text BG Players start AFTER the Image BG players
                text_bg_start_idx = 2 + self.n_bg_img
                for j in range(self.n_bg_txt):
                    if coalition[text_bg_start_idx + j] and len(self.bg_txt_groups[j]) > 0:
                        active_txt_idx_list.append(self.bg_txt_groups[j])

                if len(active_txt_idx_list) > 0:
                    active_txt_idx = torch.cat(active_txt_idx_list).unsqueeze(0)
                    tokens_orig = self.input_ids.gather(dim=1, index=active_txt_idx)
                    batch_input_ids[i : i + 1].scatter_(dim=1, index=active_txt_idx, src=tokens_orig)

            # --- C. Reshape and Qwen2-VL Fix ---
            if self.ndim == 4:
                _, C, H, W = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, C, H, W)
                batch_inputs = self.inputs

            elif self.ndim == 3:
                _, num_pixels, patch_dim = self.origin_shape

                # 1. Safely transpose the axes back
                batch_pixels = batch_feats.transpose(1, 2)
                # 2. Flatten for Qwen2-VL
                batch_pixels = batch_pixels.reshape(current_batch_size * num_pixels, patch_dim)

                # 3. Duplicate the image_grid_thw
                # batch_inputs = {k: v for k, v in self.inputs.items()}
                batch_inputs = dict(self.inputs.items())
                if "image_grid_thw" in batch_inputs:
                    grid = batch_inputs["image_grid_thw"]
                    if grid.ndim == 1:
                        batch_inputs["image_grid_thw"] = grid.unsqueeze(0).repeat(current_batch_size, 1)
                    else:
                        batch_inputs["image_grid_thw"] = grid.repeat(current_batch_size, 1)
            else:
                raise ValueError(
                    "The dimension of pixel values is not supported ! \
                        Must be (B, C, H, W) or (B, num_patches, patch_dim)"
                )

            batch_target_positions = [self.target_positions[0] for _ in range(current_batch_size)]
            batch_target_ids = self.target_ids.expand(current_batch_size, -1)

            probs = score_output(
                model=self.model,
                inputs=batch_inputs,
                input_ids=batch_input_ids,
                pixel_values=batch_pixels,
                output_ids=batch_target_ids,
                positions=batch_target_positions,
            ).numpy()

            coalitions_outputs[start_idx:end_idx] = probs

        return coalitions_outputs

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """
        Input: Coalitions as a boolean np.array of shape (n_coalitions, n_players).
        Output: Model outputs for the coalitions of shape (n_coalitions, ).
        """
        n_coalitions = coalitions.shape[0]
        coalitions_outputs = np.zeros(n_coalitions)

        # --- Clean Batching Logic (Inspired by the GitHub code) ---
        for start_idx in range(0, n_coalitions, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_coalitions)
            current_batch_size = end_idx - start_idx
            batch_coalitions = coalitions[start_idx:end_idx]

            # 1. Pre-allocate batched baselines
            batch_feats = self.feat_baseline.expand(current_batch_size, -1, -1).clone()
            batch_input_ids = self.baseline_input_ids.expand(current_batch_size, -1).clone()

            # 2. Build the inputs using your exact scatter_ logic
            for i in range(current_batch_size):
                coalition = batch_coalitions[i]

                # --- A. Image Scatter ---
                active_img_idx_list = []
                if coalition[0] and self.top_img_idx.numel() > 0:
                    active_img_idx_list.append(self.top_img_idx.squeeze(0))  # Player 0
                for j in range(self.n_background):
                    if coalition[2 + j] and len(self.bg_img_groups[j]) > 0:
                        active_img_idx_list.append(self.bg_img_groups[j])

                if len(active_img_idx_list) > 0:
                    active_img_idx = torch.cat(active_img_idx_list).unsqueeze(0)
                    active_img_idx_exp = active_img_idx.unsqueeze(1).expand(1, self.num_img_feat, -1)
                    pixels_orig = self.feat.gather(dim=2, index=active_img_idx_exp)
                    batch_feats[i : i + 1].scatter_(dim=2, index=active_img_idx_exp, src=pixels_orig)

                # --- B. Text Scatter ---
                active_txt_idx_list = []
                if coalition[1] and self.top_token_idx.numel() > 0:
                    active_txt_idx_list.append(self.top_token_idx.squeeze(0))  # Player 1
                for j in range(self.n_background):
                    if coalition[2 + j] and len(self.bg_txt_groups[j]) > 0:
                        active_txt_idx_list.append(self.bg_txt_groups[j])

                if len(active_txt_idx_list) > 0:
                    active_txt_idx = torch.cat(active_txt_idx_list).unsqueeze(0)
                    tokens_orig = self.input_ids.gather(dim=1, index=active_txt_idx)
                    batch_input_ids[i : i + 1].scatter_(dim=1, index=active_txt_idx, src=tokens_orig)
            # Create a shallow copy of inputs so we safely alter grid metadata per-batch
            # batch_inputs = {k: v for k, v in self.inputs.items()}
            batch_inputs = dict(self.inputs.items())

            # Use model_type for routing
            model_type = getattr(self, "model_type", "").lower()

            # 3. Reshape batch_feats back to original image shape
            if "llava" in model_type:
                _, C, H, W = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, C, H, W)

                # batch_inputs = self.inputs  # Standard models don't need grid trick

            elif "qwen" in model_type:
                _, num_pixels, patch_dim = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, num_pixels, patch_dim)

                # 2. Create a copy of the inputs dict so we don't permanently alter the original
                # batch_inputs = {k: v for k, v in self.inputs.items()}

                # 3. Replicate the image_grid_thw to match the number of coalitions
                if "image_grid_thw" in batch_inputs:
                    grid = batch_inputs["image_grid_thw"]
                    # If grid is (1, 3), repeat it to (5, 3)
                    batch_inputs["image_grid_thw"] = grid.repeat(current_batch_size, 1)
                
                # --- Replicate mm_token_type_ids for the batch ---
                if "mm_token_type_ids" in batch_inputs:
                    mm_ids = batch_inputs["mm_token_type_ids"]
                    # If mm_ids is (1, seq_len), repeat it to (B, seq_len)
                    batch_inputs["mm_token_type_ids"] = mm_ids.repeat(current_batch_size, 1)
                # --- QWEN2-VL SPECIFIC FIX END ---

            elif "internvl" in model_type:
                # InternVL expects 5D: (B, num_tiles, C, H, W)
                _, num_tiles, C, H, W = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, num_tiles, C, H, W)

            else:
                raise ValueError(
                    "The dimension of pixel values is not supported ! \
                        Must be (B, C, H, W) or (B, num_patches, patch_dim)"
                )

            batch_target_positions = [self.target_positions[0] for _ in range(current_batch_size)]
            batch_target_ids = self.target_ids.expand(current_batch_size, -1)

            # 4. Score Output
            probs = score_output(
                model=self.model,
                inputs=batch_inputs,  # self.inputs,
                input_ids=batch_input_ids,
                pixel_values=batch_pixels,
                output_ids=batch_target_ids,
                positions=batch_target_positions,
            ).numpy()

            coalitions_outputs[start_idx:end_idx] = probs

        return coalitions_outputs


class MicroSynergyGame(Game):
    """
    A shapiq Game interface that computes model probabilities for
    specific coalitions of Top-K features and Background chunks.
    *ATOMIC VERSION*: Image and Text players are completely decoupled.
    """

    def __init__(
        self,
        model: BaseVLMWrapper,
        inputs: dict[str, Any],
        target_ids: Tensor,
        target_positions,
        feat: Tensor,
        feat_baseline: Tensor,
        input_ids: Tensor,
        baseline_input_ids: Tensor,
        top_img_idx,
        top_token_idx,
        bg_img_groups,
        bg_txt_groups,
        num_img_feat,
        ndim: int,
        origin_shape,
        batch_size: int = 32,
    ):
        self.n_bg_img = len(bg_img_groups)
        self.n_bg_txt = len(bg_txt_groups)
        
        # Player 0: Top Image | Player 1: Top Text 
        # Players 2 to (2 + n_bg_img - 1): Image Backgrounds
        # Players (2 + n_bg_img) to End: Text Backgrounds
        self.n_players = 2 + self.n_bg_img + self.n_bg_txt

        super().__init__(n_players=self.n_players, normalize=False)

        self.model = model
        self.inputs = inputs
        self.target_ids = target_ids
        self.target_positions = target_positions

        self.feat = feat
        self.feat_baseline = feat_baseline
        self.input_ids = input_ids
        self.baseline_input_ids = baseline_input_ids

        self.top_img_idx = top_img_idx
        self.top_token_idx = top_token_idx
        self.bg_img_groups = bg_img_groups
        self.bg_txt_groups = bg_txt_groups

        self.num_img_feat = num_img_feat
        self.ndim = ndim
        self.origin_shape = origin_shape
        self.batch_size = batch_size

        self.model_type = getattr(model.model.config, "model_type", "").lower()

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        n_coalitions = coalitions.shape[0]
        coalitions_outputs = np.zeros(n_coalitions)

        for start_idx in range(0, n_coalitions, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_coalitions)
            current_batch_size = end_idx - start_idx
            batch_coalitions = coalitions[start_idx:end_idx]

            batch_feats = self.feat_baseline.expand(current_batch_size, -1, -1).clone()
            batch_input_ids = self.baseline_input_ids.expand(current_batch_size, -1).clone()

            for i in range(current_batch_size):
                coalition = batch_coalitions[i]

                # --- A. Image Scatter ---
                active_img_idx_list = []
                if coalition[0] and self.top_img_idx.numel() > 0:
                    active_img_idx_list.append(self.top_img_idx.squeeze(0))  # Player 0
                
                # Image BG Players start at index 2
                for j in range(self.n_bg_img):
                    if coalition[2 + j] and len(self.bg_img_groups[j]) > 0:
                        active_img_idx_list.append(self.bg_img_groups[j])

                if len(active_img_idx_list) > 0:
                    active_img_idx = torch.cat(active_img_idx_list).unsqueeze(0)
                    active_img_idx_exp = active_img_idx.unsqueeze(1).expand(1, self.num_img_feat, -1)
                    pixels_orig = self.feat.gather(dim=2, index=active_img_idx_exp)
                    batch_feats[i : i + 1].scatter_(dim=2, index=active_img_idx_exp, src=pixels_orig)

                # --- B. Text Scatter ---
                active_txt_idx_list = []
                if coalition[1] and self.top_token_idx.numel() > 0:
                    active_txt_idx_list.append(self.top_token_idx.squeeze(0))  # Player 1
                
                # Text BG Players start AFTER the Image BG players
                text_bg_start_idx = 2 + self.n_bg_img
                for j in range(self.n_bg_txt):
                    if coalition[text_bg_start_idx + j] and len(self.bg_txt_groups[j]) > 0:
                        active_txt_idx_list.append(self.bg_txt_groups[j])

                if len(active_txt_idx_list) > 0:
                    active_txt_idx = torch.cat(active_txt_idx_list).unsqueeze(0)
                    tokens_orig = self.input_ids.gather(dim=1, index=active_txt_idx)
                    batch_input_ids[i : i + 1].scatter_(dim=1, index=active_txt_idx, src=tokens_orig)
            
            batch_inputs = dict(self.inputs.items())
            model_type = getattr(self, "model_type", "").lower()

            if "llava" in model_type:
                _, C, H, W = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, C, H, W)

            elif "qwen" in model_type:
                _, num_pixels, patch_dim = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, num_pixels, patch_dim)
                if "image_grid_thw" in batch_inputs:
                    grid = batch_inputs["image_grid_thw"]
                    batch_inputs["image_grid_thw"] = grid.repeat(current_batch_size, 1)
                if "mm_token_type_ids" in batch_inputs:
                    mm_ids = batch_inputs["mm_token_type_ids"]
                    batch_inputs["mm_token_type_ids"] = mm_ids.repeat(current_batch_size, 1)

            elif "internvl" in model_type:
                _, num_tiles, C, H, W = self.origin_shape
                batch_pixels = batch_feats.reshape(current_batch_size, num_tiles, C, H, W)
            else:
                raise ValueError("Unsupported pixel values dimension.")

            batch_target_positions = [self.target_positions[0] for _ in range(current_batch_size)]
            batch_target_ids = self.target_ids.expand(current_batch_size, -1)

            probs = score_output(
                model=self.model,
                inputs=batch_inputs,
                input_ids=batch_input_ids,
                pixel_values=batch_pixels,
                output_ids=batch_target_ids,
                positions=batch_target_positions,
            ).numpy()

            coalitions_outputs[start_idx:end_idx] = probs

        return coalitions_outputs


def eval_macro_sii_with_class(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    target_ids: Tensor,
    pixel_attribution: Tensor,
    token_attribution: Tensor,
    perturbation_steps: list,
    pad_token_id: int,
    special_token_ids: list,
    # target_positions: list,
    filter_keywords: bool = True,
    blur_baseline: Tensor | None = None,
    semantic_mask: Tensor | None = None,
    mask_value: float = 0.0,
    n_background_groups: int = 10,
    shapiq_budget: int = 300,
    batch_size: int = 32,
):
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

    ndim = pixel_values.ndim
    origin_shape = pixel_values.shape

    # Setup Baselines & Flattening
    feat, num_pixels = _reshape_pixels_faithfulness(
        pixel_values=pixel_values, origin_shape=origin_shape, model_type=model_type
    )

    num_img_feat = feat.shape[1]

    # Baseline image (blur)
    if blur_baseline is None:
        # blur_baseline = torch.full_like(pixel_values, mask_value).to(device)
        blur_baseline = make_blur_baseline(pixel_values=pixel_values, model_type=model_type)
    feat_baseline = blur_baseline.clone().reshape(feat.shape)
    # sal_flat_img = pixel_attribution.reshape(1, -1)

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
            valid_mask &= input_ids != skip_id

    # Count how many actual text tokens we have per batch
    num_valid_tokens = int(valid_mask.sum(dim=1).min().item())

    # --- Mask Attribution Scores ---
    masked_attribution = token_attribution.clone()
    # Apply huge negative value where mask is False (Visual tokens)
    masked_attribution[~valid_mask] = -float("inf")

    # --- Baselines ---
    baseline_input_ids = input_ids.clone()
    baseline_input_ids[valid_mask] = pad_token_id

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

    sii_curve = []

    # --- 2. Loop over K Steps ---
    for step in perturbation_steps:
        k_img = max(0, min(int(round(step * num_pixels)), num_pixels))
        k_txt = max(0, min(int(round(step * num_valid_tokens)), num_valid_tokens))

        top_img_idx = torch.empty((1, 0), dtype=torch.long, device=device)
        top_token_idx = torch.empty((1, 0), dtype=torch.long, device=device)

        if k_img > 0:
            _, top_img_idx = torch.topk(sal_flat_img, k_img, dim=-1)
        if k_txt > 0:
            _, top_token_idx = torch.topk(masked_attribution, k_txt, dim=-1)

        top_img_idx = top_img_idx.to(device)
        top_token_idx = top_token_idx.to(device)

        # Background chunking
        is_bg_img = torch.ones((1, num_pixels), dtype=torch.bool, device=device)
        is_bg_img.scatter_(1, top_img_idx, False)
        bg_img_indices = is_bg_img.nonzero(as_tuple=True)[1][torch.randperm(num_pixels - k_img)]

        is_bg_txt = valid_mask.clone()
        is_bg_txt.scatter_(1, top_token_idx, False)
        bg_txt_indices = is_bg_txt.nonzero(as_tuple=True)[1][torch.randperm(num_valid_tokens - k_txt)]

        bg_img_groups = torch.tensor_split(bg_img_indices, n_background_groups)
        bg_txt_groups = torch.tensor_split(bg_txt_indices, n_background_groups)

        # --- 3. Instantiate Game and Run Shapiq ---
        game = MacroSynergyGame(
            model=model,
            inputs=inputs,
            target_ids=target_ids,
            target_positions=target_positions,
            feat=feat,
            feat_baseline=feat_baseline,
            input_ids=input_ids,
            baseline_input_ids=baseline_input_ids,
            top_img_idx=top_img_idx,
            top_token_idx=top_token_idx,
            bg_img_groups=bg_img_groups,
            bg_txt_groups=bg_txt_groups,
            num_img_feat=num_img_feat,
            ndim=ndim,
            origin_shape=origin_shape,
            batch_size=batch_size,
        )
        print(game.n_players)

        approximator = shapiq.approximator.SHAPIQ(n=game.n_players, max_order=2, index="SII")
        interaction_values = approximator.approximate(budget=shapiq_budget, game=game)

        sii_curve.append(interaction_values[(0, 1)])

    return {
        "sii_curve": sii_curve,
        "sii_auc": np.trapezoid(sii_curve, x=perturbation_steps),
    }

def eval_exact_sii_with_class(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    target_ids: Tensor,
    pixel_attribution: Tensor,
    token_attribution: Tensor,
    perturbation_steps: list,
    pad_token_id: int,
    special_token_ids: list,
    filter_keywords: bool = True,
    blur_baseline: Tensor | None = None,
    semantic_mask: Tensor | None = None,
    mask_value: float = 0.0,
    batch_size: int = 64,
    shapiq_budget: int | None = None,
    # Removed n_background_groups and shapiq_budget as they are now forced to exact math
):
    device = model.device
    pixel_values = inputs["pixel_values"].to(device)
    pixel_attribution = pixel_attribution.to(device)
    input_ids = inputs["input_ids"].to(device)
    target_ids = target_ids.to(device)

    model_type = getattr(model.model.config, "model_type", "").lower()

    if "internvl" in model_type:
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
    elif "qwen" in model_type:
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)
    else:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

    ndim = pixel_values.ndim
    origin_shape = pixel_values.shape

    feat, num_pixels = _reshape_pixels_faithfulness(
        pixel_values=pixel_values, origin_shape=origin_shape, model_type=model_type
    )
    num_img_feat = feat.shape[1]

    if blur_baseline is None:
        blur_baseline = make_blur_baseline(pixel_values=pixel_values, model_type=model_type)
    feat_baseline = blur_baseline.clone().reshape(feat.shape)

    B, *_ = origin_shape
    if pixel_attribution.ndim == 4 or pixel_attribution.ndim == 3:
        sal_flat_img = pixel_attribution.reshape(B, -1)
    elif pixel_attribution.ndim == 2:
        sal_flat_img = pixel_attribution
    else:
        raise ValueError("pixel_attribution must be 2D, 3D, or 4D.")

    if semantic_mask is not None:
        if semantic_mask.ndim == 1:
            semantic_mask = semantic_mask.unsqueeze(0)
        valid_mask = semantic_mask.to(device).clone()
    else:
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)

    if special_token_ids is not None:
        for skip_id in special_token_ids:
            valid_mask &= input_ids != skip_id

    num_valid_tokens = int(valid_mask.sum(dim=1).min().item())

    masked_attribution = token_attribution.clone()
    masked_attribution[~valid_mask] = -float("inf")

    baseline_input_ids = input_ids.clone()
    baseline_input_ids[valid_mask] = pad_token_id

    if filter_keywords:
        target_positions = get_most_important_tokens_multimodal(
            model, inputs, input_ids, baseline_input_ids, pixel_values, blur_baseline, target_ids,
        )
    else:
        seq_len = target_ids.shape[1]
        target_positions = [torch.arange(seq_len, device=device) for _ in range(B)]

    sii_curve = []

    for step in perturbation_steps:
        k_img = max(0, min(int(round(step * num_pixels)), num_pixels))
        k_txt = max(0, min(int(round(step * num_valid_tokens)), num_valid_tokens))

        top_img_idx = torch.empty((1, 0), dtype=torch.long, device=device)
        top_token_idx = torch.empty((1, 0), dtype=torch.long, device=device)

        if k_img > 0:
            _, top_img_idx = torch.topk(sal_flat_img, k_img, dim=-1)
        if k_txt > 0:
            _, top_token_idx = torch.topk(masked_attribution, k_txt, dim=-1)

        top_img_idx = top_img_idx.to(device)
        top_token_idx = top_token_idx.to(device)

        is_bg_img = torch.ones((1, num_pixels), dtype=torch.bool, device=device)
        is_bg_img.scatter_(1, top_img_idx, False)
        bg_img_indices = is_bg_img.nonzero(as_tuple=True)[1]

        is_bg_txt = valid_mask.clone()
        is_bg_txt.scatter_(1, top_token_idx, False)
        bg_txt_indices = is_bg_txt.nonzero(as_tuple=True)[1]

        # ---------------------------------------------------------
        # TRUE ATOMIC MODE: Every index is its own independent group
        # ---------------------------------------------------------
        bg_img_groups = [torch.tensor([idx], device=device) for idx in bg_img_indices]
        bg_txt_groups = [torch.tensor([idx], device=device) for idx in bg_txt_indices]

        print(len(bg_img_groups))
        print(len(bg_txt_groups))

        game = MicroSynergyGame(
            model=model,
            inputs=inputs,
            target_ids=target_ids,
            target_positions=target_positions,
            feat=feat,
            feat_baseline=feat_baseline,
            input_ids=input_ids,
            baseline_input_ids=baseline_input_ids,
            top_img_idx=top_img_idx,
            top_token_idx=top_token_idx,
            bg_img_groups=bg_img_groups,
            bg_txt_groups=bg_txt_groups,
            num_img_feat=num_img_feat,
            ndim=ndim,
            origin_shape=origin_shape,
            batch_size=batch_size,
        )
        
        # ---------------------------------------------------------
        # FORCE EXACT BUDGET (2^N)
        # ---------------------------------------------------------
        if shapiq_budget is None:
            shapiq_budget = 2 ** game.n_players
        
     
        print(f"[*] Running exact atomic game with {game.n_players} players. Budget = {shapiq_budget}")

        approximator = shapiq.approximator.SHAPIQ(n=game.n_players, max_order=2, index="SII")
        interaction_values = approximator.approximate(budget=shapiq_budget, game=game)

        sii_curve.append(interaction_values[(0, 1)])

    return {
        "sii_curve": sii_curve,
        "sii_auc": np.trapezoid(sii_curve, x=perturbation_steps),
    }
