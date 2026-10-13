import json
import math
import os
import re
import string

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor


def align_llm_visuals_to_pixels(
    pixel_attribution: Tensor,
    inputs: dict,
    config,
) -> Tensor:
    """
    Reshapes and interpolates LLM-level pixel attributions to match
    the exact spatial footprint of the model's original pixel_values.

    Args:
        pixel_attribution: Tensor of shape [gen_len, num_llm_tokens]
        inputs: The original Hugging Face inputs dictionary.

    Returns:
        Tensor formatted specifically for the perturbation metrics:
        - InternVL: [gen_len, num_tiles, H, W]
        - Standard: [gen_len, H, W]
        - QwenVL:   [gen_len, num_patches]
    """
    pixel_values = inputs.get("pixel_values")

    # Safety check: if there is no image, just return the raw tensor
    if pixel_values is None or pixel_attribution.numel() == 0:
        return pixel_attribution

    gen_len, num_llm_tokens = pixel_attribution.shape
    # ndim = pixel_values.ndim
    model_type = config.model_type

    # ---------------------------------------------------------
    # CASE C: InternVL (5D AnyRes Tiling)
    # Shape: (Batch, num_tiles, C, H, W)
    # ---------------------------------------------------------
    if "internvl" in model_type:
        num_tiles, C, target_h, target_w = pixel_values.shape

        # InternVL produces a 16x16 LLM feature map per tile (256 tokens)
        tile_size = getattr(config, "image_seq_length", 256)
        llm_grid_h = llm_grid_w = int(math.sqrt(tile_size))

        # 1. Reshape to allow 2D spatial interpolation across all tiles simultaneously
        # Shape: [gen_len * num_tiles, 1, 16, 16]
        pixel_attr_2d = pixel_attribution.view(gen_len * num_tiles, 1, llm_grid_h, llm_grid_w)

        # 2. Upscale from 16x16 to target size (usually 448x448)
        pixel_attr_upscaled = F.interpolate(
            pixel_attr_2d,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # 3. Reshape strictly to the 4D format your metrics expect
        # Shape: [gen_len, num_tiles, target_h, target_w]
        return pixel_attr_upscaled.view(gen_len, num_tiles, target_h, target_w)

    # ---------------------------------------------------------
    # CASE A: Standard VLM (e.g., LLaVA) (4D)
    # Shape: (Batch, C, H, W)
    # ---------------------------------------------------------
    elif "llava" in model_type:
        _, C, target_h, target_w = pixel_values.shape

        # Standard models usually have square ViT grids
        llm_grid_h = llm_grid_w = int(math.sqrt(num_llm_tokens))

        pixel_attr_2d = pixel_attribution.view(gen_len, 1, llm_grid_h, llm_grid_w)

        pixel_attr_upscaled = F.interpolate(
            pixel_attr_2d,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Shape: [gen_len, target_h, target_w]
        return pixel_attr_upscaled.view(gen_len, target_h, target_w)

    # ---------------------------------------------------------
    # CASE B: Packed Patches VLM (e.g., Qwen-VL) (3D)
    # Shape: (Batch, num_patches, patch_dim)
    # ---------------------------------------------------------
    elif "qwen" in model_type:
        _, target_h, target_w = inputs["image_grid_thw"][0].cpu().numpy().tolist()

        # Qwen uses a 2x2 spatial merge before the LLM
        spatial_merge_size = getattr(config, "spatial_merge_size", 2)
        llm_grid_h = target_h // spatial_merge_size
        llm_grid_w = target_w // spatial_merge_size

        pixel_attr_2d = pixel_attribution.view(gen_len, 1, llm_grid_h, llm_grid_w)

        pixel_attr_upscaled = F.interpolate(
            pixel_attr_2d,
            size=(target_h, target_w),
            mode="nearest",  # Nearest is better for Qwen's discrete patch structure
        )

        # Qwen metrics expect flattened patches: [gen_len, target_h * target_w]
        return pixel_attr_upscaled.view(gen_len, -1)

    else:
        raise ValueError("Could not infer spatial grid from inputs. Unknown architecture.")


def align_attribution_to_patches(high_res_attr: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Downsamples a pixel-level attribution map to match the VLM's patch sequence.

    Args:
        high_res_attr: Shape (num_answer_tokens, H, W) or (H, W). The raw attribution.
        image_grid_thw: Shape (1, 3). The temporal/height/width grid from Qwen.

    Returns:
        patch_attr: Shape (num_answer_tokens, num_patches).
                    Ready for your faithfulness metric!
    """
    # 1. Handle dimensionality
    if high_res_attr.dim() == 2:
        high_res_attr = high_res_attr.unsqueeze(0)  # (1, H, W)

    num_tokens = high_res_attr.shape[0]

    # 2. Extract the target grid dimensions from Qwen's metadata
    # image_grid_thw is usually [temporal (1), grid_h, grid_w]
    grid_h = int(image_grid_thw[0, 1].item())
    grid_w = int(image_grid_thw[0, 2].item())
    num_patches = int(grid_h * grid_w)

    # 3. Add dummy channel dim for PyTorch pooling functions: (N, C, H, W)
    attr_4d = high_res_attr.unsqueeze(1).float()

    # 4. Downsample to the Patch Grid
    # Adaptive Average Pooling is perfect here because it guarantees the output
    # will be exactly (grid_h, grid_w), regardless of the input (H, W).
    # It effectively asks: "What is the average importance of the pixels inside this patch?"
    patch_grid_attr = F.adaptive_avg_pool2d(attr_4d, output_size=(grid_h, grid_w))

    # 5. Flatten back to the sequence format Qwen expects
    # (N, 1, grid_h, grid_w) -> (N, grid_h * grid_w) -> (N, num_patches)
    patch_attr_flat = patch_grid_attr.view(num_tokens, num_patches)

    return patch_attr_flat


def create_semantic_mask_robust_(input_ids: Tensor, processor, prefix_text: str, core_question: str) -> Tensor:
    """
    Finds the exact tokens corresponding to the core question using cumulative decoding.
    This safely ignores BPE space-merging and special inserted tokens.

    Args:
        input_ids: (1, Seq_Len) tensor
        processor: The model's processor/tokenizer
        prefix_text: The instruction text BEFORE the question (e.g., "Answer strictly...")
        core_question: The question itself
    """
    seq_len = input_ids.shape[1]
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=input_ids.device)

    # Clean strings to avoid trailing whitespace mismatches
    prefix_clean = prefix_text.strip()
    target_clean = (prefix_text + core_question).strip()

    start_idx = None
    end_idx = None

    for i in range(1, seq_len + 1):
        # Decode the sequence up to the current token
        # skip_special_tokens=True ignores <|im_start|>, <|image_pad|>, etc.
        decoded_so_far = processor.decode(input_ids[0, :i], skip_special_tokens=True)

        # --- 1. Find where the Question Starts ---
        if start_idx is None and prefix_clean in decoded_so_far:
            # Check if the current token ended EXACTLY at the prefix.
            start_idx = i if decoded_so_far.endswith(prefix_clean) else i - 1
            # The NEXT token (index i) starts the question

        # --- 2. Find where the Question Ends ---
        if start_idx is not None and target_clean in decoded_so_far:
            end_idx = i - 1
            break

    if start_idx is not None and end_idx is not None:
        mask[0, start_idx : end_idx + 1] = True
    else:
        print(f"[!] Warning: Could not align semantic mask for: '{core_question}'")
        # Fallback: mask all text (or handle as needed)

    return mask


def create_semantic_mask_robust(input_ids: torch.Tensor, processor, core_question: str) -> torch.Tensor:
    """
    Finds the exact tokens corresponding to the core question by searching for the
    tightest token window that decodes to contain the question.
    This safely ignores BPE space-merging and hidden chat template tags.
    """
    seq_len = input_ids.shape[1]
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=input_ids.device)

    target_clean = core_question.strip()

    best_i = None
    best_j = None
    min_window = float("inf")

    # Brute force search: O(N^2) but extremely fast for short VQA sequences
    for i in range(seq_len):
        for j in range(i + 1, seq_len + 1):
            # Decode the current slice
            decoded = processor.decode(input_ids[0, i:j], skip_special_tokens=True)

            # Does this slice contain our full question?
            if target_clean in decoded:
                window_size = j - i

                # We want the tightest possible bounds around the question
                if window_size < min_window:
                    min_window = window_size
                    best_i = i
                    best_j = j

                # Once we find it for this 'i', making 'j' larger just adds trailing
                # context we don't want, so we break and move to the next 'i'
                break

    if best_i is not None and best_j is not None:
        mask[0, best_i:best_j] = True
    else:
        print(f"[!] Warning: Could not align semantic mask for: '{core_question}'")
        # Fallback: Mask all non-special tokens
        mask = torch.ones_like(mask)

    return mask


def get_processed_indices(output_file: str, total_dataset_len: int, max_samples: int | None = None) -> set:
    """
    Scans the output JSONL file to find indices of already processed samples.
    Returns a set of completed sample indices for resume logic.
    """
    processed_indices = set()

    if os.path.exists(output_file):
        print("[*] Found existing results file. Scanning for completed samples...")
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "sample_idx" in data:
                            processed_indices.add(data["sample_idx"])
                    except json.JSONDecodeError:
                        pass

        # Calculate the target denominator for the print statement
        total_samples = total_dataset_len
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)

        print(f"[*] Skipping {len(processed_indices)} / {total_samples} already processed samples.")

    return processed_indices


def save_to_jsonl(data: dict, filepath: str):
    """Appends a dictionary as a JSON line to a file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def load_yaml(file_path):
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_ynvqa_token_index(new_ids, text_answer, tokenizer) -> int | None:
    """
    Finds the exact index of the 'yes' or 'no' token in the generated sequence.
    """
    text_lower = text_answer.lower()

    # Fast check: If the model didn't even say yes or no, return None immediately
    if "yes" not in text_lower and "no" not in text_lower:
        return None

    # Ensure new_ids is a flat 1D list/tensor
    if new_ids.dim() > 1:
        new_ids = new_ids[0]

    # Find the exact token index
    for idx, tok_id in enumerate(new_ids):
        # Decode just this single token
        word = tokenizer.decode(tok_id).strip().lower()
        # Remove punctuation like 'yes.' or 'no,'
        word = word.translate(str.maketrans("", "", string.punctuation))

        if word in ["yes", "no"]:
            return idx  # Found it! Return as a list for target_indices

    return None  # Fallback


def find_mcvqa_token_index(new_ids, tokenizer, choices=None):
    """
    Scans tokenized output and scores candidates.
    Includes Math penalties, Special Token boundaries for LLaVA/Qwen, and Empty-Token detection.
    """
    if choices is None:
        choices = ["a", "b", "c", "d"]

    if hasattr(new_ids, "dim") and new_ids.dim() > 1:
        new_ids = new_ids[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(new_ids)
    lower_choices = [str(c).lower() for c in choices]

    candidates = []

    for idx, token in enumerate(tokens):
        # re.sub automatically strips out '▁', 'Ġ', and ' ' to reveal the pure alphanumeric letter
        clean_token = token.lower()
        raw_chars = re.sub(r"[^a-z0-9]", "", clean_token)

        if len(raw_chars) == 1 and raw_chars in lower_choices:
            score = 0

            # 1. Self-Context
            if any(p in token for p in ["(", ")", ".", ":"]):
                score += 2

            # 2. Previous Context
            if idx > 0:
                prev_token = tokens[idx - 1].lower()
                # Clean the previous token to see if it's literally just empty space or special chars
                prev_clean = re.sub(r"[^a-z0-9]", "", prev_token)

                # BOOST: If previous token is just spaces OR a structural/special start token
                if not prev_clean or any(
                    w in prev_token
                    for w in ["answer", "is", "option", ":", "(", "\n", "<s>", "<bos>", "<pad>", "<|im_start|>"]
                ):
                    score += 3

                # MATH PENALTY
                if any(m in prev_token for m in ["+", "-", "=", "^", "*", "/", "\\", "{", "}"]):
                    score -= 10
            else:
                score += 5  # First token boost

            # 3. Next Context
            if idx < len(tokens) - 1:
                next_token = tokens[idx + 1].lower()
                next_clean = re.sub(r"[^a-z0-9]", "", next_token)

                # BOOST: If next token is just spaces OR a structural/special end token
                if not next_clean or any(
                    w in next_token for w in [")", ".", "\n", ":", "</s>", "<eos>", "<pad>", "<|im_end|>"]
                ):
                    score += 3

                # MATH PENALTY
                if any(m in next_token for m in ["+", "-", "=", "^", "*", "/", "\\", "{", "}"]):
                    score -= 10
            else:
                score += 3  # Last token boost

            # 4. Penalty for normal words hidden in a sentence
            if score <= 0:
                score -= 5

            candidates.append((score, idx, token))

    # Evaluate the candidates
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_score, best_idx, best_token = candidates[0]

        # ACCEPT IF: It survived the penalties OR it's the ONLY valid letter the model generated!
        if best_score > -1 or len(candidates) == 1:
            return best_idx, best_token

    # Fallback
    return -1, tokens[0] if tokens else ""
