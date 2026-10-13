import re
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.models import BaseVLMWrapper


def score_output(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    output_ids: torch.Tensor,
    positions: list[torch.Tensor],
) -> torch.Tensor:
    generated_ids = torch.cat((input_ids, output_ids), dim=1)

    probs_pred = pred_probs(
        model,
        inputs,
        generated_ids,
        pixel_values,
        output_ids,
    )
    # Fix
    batch_size = probs_pred.shape[0]
    device = probs_pred.device

    # FIX: Vectorized position scoring
    # Pre-allocate the result tensor on the GPU
    batch_scores = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for b in range(batch_size):
        p = positions[b].to(device)
        if len(p) > 0:
            # Gather and mean directly on the GPU, much faster than appending to a list
            # batch_scores[b] = probs_pred[b, p].sum()
            # Normalize by length to get the average log probability per token
            log_prob_sum = probs_pred[b, p].sum()

            # 2. Convert back to raw probability space [0, 1] for Game Theory Math!
            batch_scores[b] = torch.exp(log_prob_sum)
            # batch_scores[b] = log_prob_sum

    scores = batch_scores.cpu()

    return scores


def pred_probs(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    new_input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None,
    output_ids: torch.Tensor,
) -> torch.Tensor:

    device = model.device

    # attention_mask = torch.ones_like(new_input_ids).to(device)
    pad_token_id = model.processor.tokenizer.pad_token_id if model.processor.tokenizer.pad_token_id is not None else 0
    attention_mask = (new_input_ids != pad_token_id).long().to(device)

    other_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

    # ==========================================================
    # MULTIMODAL ROPE FIX FOR QWEN 3.5
    # ==========================================================
    if "mm_token_type_ids" in other_kwargs:
        orig_mm_types = other_kwargs["mm_token_type_ids"]
        seq_len_diff = new_input_ids.shape[1] - orig_mm_types.shape[1]
        
        # If we concatenated generated tokens, pad the type array with 0 (text type)
        if seq_len_diff > 0:
            pad_types = torch.zeros(
                (orig_mm_types.shape[0], seq_len_diff), 
                dtype=orig_mm_types.dtype, 
                device=orig_mm_types.device
            )
            other_kwargs["mm_token_type_ids"] = torch.cat([orig_mm_types, pad_types], dim=1)
            
    # Always delete position_ids if modifying sequence length so Qwen 
    # is forced to dynamically recalculate RoPE indices correctly!
    if "position_ids" in other_kwargs:
        del other_kwargs["position_ids"]
    # ==========================================================

    # Intercept and flatten 5D tensors for InternVL/AnyRes models
    if pixel_values is not None and pixel_values.ndim == 5:
        B, num_tiles, C, H, W = pixel_values.shape
        # Flatten (B, num_tiles) into a single batch dimension for the Vision Tower
        pixel_values = pixel_values.view(B * num_tiles, C, H, W)

    with torch.no_grad():
        outputs = model.model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            # return_probs=False,
            **other_kwargs,
        )  # [batch_size, seq_len, vocab_size]
    all_logits = outputs.logits

    returned_logits = all_logits[
        :, -output_ids.shape[-1] - 1 : -1, :
    ]  # The reason for the minus 1 is that the generated content is in the previous position
    # returned_logits = F.softmax(returned_logits, dim=-1) # (batch_size, selected_tokens, vocab_size)
    returned_logits = F.log_softmax(returned_logits, dim=-1)  # (batch_size, selected_tokens, vocab_size)

    returned_logits = returned_logits.gather(dim=2, index=output_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (batch_size, selected_tokens)
    return returned_logits


def get_most_important_tokens_pixel(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    blur_baseline: torch.Tensor,
    # device: torch.device,
) -> list[torch.Tensor]:
    # Run keyword filtering logic (Batched simplified version of find_keywords)
    # We need probabilities for Original vs Blur
    target_positions = []
    B = input_ids.shape[0]
    with torch.no_grad():
        # Combine prompt + answer
        full_input_ids = torch.cat((input_ids, target_ids), dim=1)

        # Forward Original
        # out_orig = model(pixel_values=pixel_values, input_ids=full_input_ids)
        # logits_orig = out_orig.logits[:, input_ids.shape[1]-1 : -1, :] # Shift for next-token prediction
        # probs_orig = torch.gather(logits_orig.softmax(-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
        probs_orig = pred_probs(
            model=model,
            inputs=inputs,
            new_input_ids=full_input_ids,
            pixel_values=pixel_values,
            output_ids=target_ids,
        )

        # Forward Blur
        # out_blur = model(pixel_values=blur_baseline.to(device), input_ids=full_input_ids)
        # logits_blur = out_blur.logits[:, input_ids.shape[1]-1 : -1, :]
        # probs_blur = torch.gather(logits_blur.softmax(-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
        probs_blur = pred_probs(
            model=model,
            inputs=inputs,
            new_input_ids=full_input_ids,
            pixel_values=blur_baseline,
            output_ids=target_ids,
        )

        # Ratio Check: Log(P_orig / P_blur) > 1.0
        # Corresponds to: condition = (torch.log(probs)-torch.log(probs_blur) > 1.0)
        ratio_mask = (torch.log(probs_orig + 1e-9) - torch.log(probs_blur + 1e-9)) > 1.0

        for b in range(B):
            # Get indices where condition is true
            valid_indices = torch.where(ratio_mask[b])[0]
            if len(valid_indices) == 0:
                # Fallback: Pick max drop if no strong keywords found
                valid_indices = torch.argmax(probs_orig[b] - probs_blur[b]).unsqueeze(0)
            target_positions.append(valid_indices)
    return target_positions


def get_most_important_tokens_token(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    input_ids: torch.Tensor,
    baseline_input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> list[torch.Tensor]:
    # Adapted logic for text
    pixel_values = inputs.get("pixel_values")

    with torch.no_grad():
        # Score Original
        full_orig = torch.cat((input_ids, target_ids), dim=1)
        probs_orig = pred_probs(
            model=model,
            inputs=inputs,
            new_input_ids=full_orig,
            pixel_values=pixel_values,
            output_ids=target_ids,
        )

        # Score Baseline (Empty Prompt)
        full_base = torch.cat((baseline_input_ids, target_ids), dim=1)
        probs_base = pred_probs(
            model=model,
            inputs=inputs,
            new_input_ids=full_base,
            pixel_values=pixel_values,
            output_ids=target_ids,
        )

        # Calculate Ratio
        target_positions = []
        B = input_ids.shape[0]
        ratio_mask = (torch.log(probs_orig + 1e-9) - torch.log(probs_base + 1e-9)) > 1.0

        for b in range(B):
            valid_indices = torch.where(ratio_mask[b])[0]
            if len(valid_indices) == 0:
                valid_indices = torch.argmax(probs_orig[b] - probs_base[b]).unsqueeze(0)
            target_positions.append(valid_indices)

    return target_positions


def get_most_important_tokens_multimodal(
    model: BaseVLMWrapper,
    inputs: dict[str, Any],
    input_ids: torch.Tensor,
    base_ids: torch.Tensor,
    pixel_val: torch.Tensor,
    base_pix: torch.Tensor,
    target_ids: torch.Tensor,
) -> list[torch.Tensor]:
    # Determine keywords based on Joint Original vs Joint Baseline
    with torch.no_grad():
        full_orig = torch.cat((input_ids, target_ids), dim=1)
        full_base = torch.cat((base_ids, target_ids), dim=1)

        # Joint Original
        probs_orig = pred_probs(
            model=model,
            inputs=inputs,
            new_input_ids=full_orig,
            pixel_values=pixel_val,
            output_ids=target_ids,
        )
        # Joint Baseline (Blur + Pad)
        probs_base = pred_probs(
            model=model,
            inputs=inputs,
            new_input_ids=full_base,
            pixel_values=base_pix,
            output_ids=target_ids,
        )

        # Calculate Ratio
        target_positions = []
        B = input_ids.shape[0]
        ratio_mask = (torch.log(probs_orig + 1e-9) - torch.log(probs_base + 1e-9)) > 1.0

        for b in range(B):
            valid_indices = torch.where(ratio_mask[b])[0]
            if len(valid_indices) == 0:
                valid_indices = torch.argmax(probs_orig[b] - probs_base[b]).unsqueeze(0)
            target_positions.append(valid_indices)

    return target_positions


def make_blur_baseline(pixel_values: torch.Tensor, model_type: str) -> torch.Tensor:
    if "internvl" in model_type:
        # (B, num_tiles, C, H, W) -> Reshape, Blur, Reshape Back
        B, N, C, H, W = pixel_values.shape
        flat_tiles = pixel_values.view(B * N, C, H, W)
        blurred_tiles = TF.gaussian_blur(flat_tiles, kernel_size=[51, 51], sigma=[10.0, 10.0])
        blur_baseline = blurred_tiles.view(B, N, C, H, W)

    elif "llava" in model_type:
        # (B, C, H, W) -> Direct Gaussian Blur
        blur_baseline = TF.gaussian_blur(pixel_values, kernel_size=[51, 51], sigma=[10.0, 10.0])

    elif "qwen" in model_type:
        # (B, num_patches, patch_dim) -> Noisy Zeros (Mean + Variance)
        # Spatial geometry is flattened, so we use the variance-safe zero baseline
        blur_baseline = torch.zeros_like(pixel_values) + (torch.randn_like(pixel_values) * 0.1)

    else:
        raise ValueError(
            f"Model: {model_type} not implemented ! \
                        Pixel_values shape must be 3D, 4D, or 5D."
        )
    return blur_baseline


def _reshape_pixels_faithfulness(pixel_values: torch.Tensor, origin_shape, model_type: str):
    # Setup Baselines & Flattening
    # if ndim == 5: # INTERNVL: (B, num_tiles, C, H, W)
    if "internvl" in model_type:  # INTERNVL: (B, num_tiles, C, H, W)
        B, num_tiles, C, H, W = origin_shape
        num_pixels = num_tiles * H * W
        feat = pixel_values.reshape(B, C, num_pixels)
    # elif ndim == 4: # STANDARD: (B, C, H, W)
    elif "llava" in model_type:
        B, C, H, W = origin_shape
        num_pixels = H * W
        feat = pixel_values.reshape(B, C, num_pixels)
    # elif ndim == 3: # QWENVL: (B, num_patches, patch_dim)
    elif "qwen" in model_type:
        B, num_pixels, patch_dim = origin_shape
        feat = pixel_values.reshape(B, patch_dim, num_pixels)
    else:
        raise ValueError(
            f"Model: {model_type} not implemented ! \
                        Pixel_values shape must be 3D, 4D, or 5D."
        )
    return feat, num_pixels


def _reshape_pixels_back_faithfulness(feat_pert: torch.Tensor, origin_shape, model_type: str):
    if "internvl" in model_type:
        B, num_tiles, C, H, W = origin_shape
        pert_pixels = feat_pert.reshape(B, num_tiles, C, H, W)
    # elif ndim == 4:
    elif "llava" in model_type:
        B, C, H, W = origin_shape
        pert_pixels = feat_pert.reshape(B, C, H, W)
    # elif ndim == 3:
    elif "qwen" in model_type:
        B, num_pixels, patch_dim = origin_shape
        pert_pixels = feat_pert.reshape(B, num_pixels, patch_dim)
    else:
        raise ValueError(
            f"Model: {model_type} not implemented ! \
                        Pixel_values shape must be 3D, 4D, or 5D."
        )
    return pert_pixels


def get_text_mask(input_ids: torch.Tensor, model_type: str, tokenizer) -> torch.Tensor:
    """
    Blazing fast, model-specific mask generator.
    Finds the exact tokens between the 'User' and 'Assistant' anchors using pure tensor math.
    """
    if input_ids.dim() > 1:
        input_ids = input_ids.squeeze()

    seq_len = input_ids.shape[0]
    valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)

    # 1. Define Model-Specific Anchor Strings
    # We use exactly what the processor injects.
    model_type = model_type.lower()
    if "llava" in model_type:
        user_str = "USER:"
        asst_str = "ASSISTANT:"
    elif "qwen" in model_type or "internvl" in model_type:
        # Based on your logs, Qwen and InternVL both use these plain words
        user_str = "user"
        asst_str = "assistant"
    else:
        raise ValueError(f"Model type '{model_type}' not supported in fast-mask.")

    # 2. Get the exact Token IDs for these anchors
    # We do this once per batch. It correctly handles LLaVA splitting "USER:" into ["US", "ER", ":"]
    user_ids = torch.tensor(tokenizer.encode(user_str, add_special_tokens=False), device=input_ids.device)
    asst_ids = torch.tensor(tokenizer.encode(asst_str, add_special_tokens=False), device=input_ids.device)

    user_len = len(user_ids)
    asst_len = len(asst_ids)

    # 3. Fast Tensor Search (No Strings!)
    # We slide over the tensor to find the exact match for the anchor sequences
    start_idx = -1
    end_idx = -1

    # Find where the 'User' anchor ends
    for i in range(seq_len - user_len + 1):
        if torch.equal(input_ids[i : i + user_len], user_ids):
            start_idx = i + user_len  # The question starts IMMEDIATELY AFTER the user anchor
            break

    # Find where the 'Assistant' anchor begins (Search from start_idx onwards to save time)
    if start_idx != -1:
        for i in range(start_idx, seq_len - asst_len + 1):
            if torch.equal(input_ids[i : i + asst_len], asst_ids):
                end_idx = i  # The question ends IMMEDIATELY BEFORE the assistant anchor
                break

    # 4. Apply the Mask
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        valid_mask[start_idx:end_idx] = True
    else:
        print(f"\n[!] Fast Mask Failed for {model_type}. Anchors not found in tensor.")
        print("[!] Falling back to all True.")
        valid_mask[:] = True

    return valid_mask


def find_decision_token_index(new_ids, tokenizer, choices=None):
    """
    Scans tokenized output and scores candidates based on their surrounding context.
    Perfectly handles:
    - "There is a bag. Answer: (a)"
    - "Answer: a) there is a bag"
    - "(b) Closed"
    """
    if choices is None:
        choices = ["a", "b", "c", "d"]

    tokens = tokenizer.convert_ids_to_tokens(new_ids)
    lower_choices = [str(c).lower() for c in choices]

    candidates = []

    for idx, token in enumerate(tokens):
        # Clean token for core letter extraction
        clean_token = token.replace("Ġ", "").replace(" ", "").lower()
        raw_chars = re.sub(r"[^a-z0-9]", "", clean_token)

        # If we found a standalone letter that matches our choices...
        if len(raw_chars) == 1 and raw_chars in lower_choices:
            score = 0

            # 1. Self-Context: Does the token itself contain punctuation? e.g., "(a" or "a)"
            if "(" in token or ")" in token or "." in token:
                score += 2

            # 2. Previous Context: What came right before it?
            if idx > 0:
                prev_token = tokens[idx - 1].lower()
                # If preceded by "answer", ":", "(", or a newline, massive boost!
                if "answer" in prev_token or ":" in prev_token or "(" in prev_token or "\n" in prev_token:
                    score += 3

            # 3. Next Context: What comes right after it?
            if idx < len(tokens) - 1:
                next_token = tokens[idx + 1].lower()
                # If followed by ")", ".", or a newline, massive boost!
                if ")" in next_token or "." in next_token or "\n" in next_token:
                    score += 3

            # 4. Penalty: If it has ZERO structural neighbors, it's probably the word "a" in a sentence.
            if score == 0:
                score -= 5

            candidates.append((score, idx, token))

    # Evaluate the candidates
    if candidates:
        # Sort by highest score first
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_token = candidates[0]

        # As long as the best candidate isn't heavily penalized, return it!
        if best_score > -1:
            return best_idx, best_token

    # Fallback: If it completely hallucinated and we found nothing valid
    return 0, tokens[0]
