from dataclasses import asdict, dataclass
import re
import unicodedata
import cv2
import numpy as np
import torch

@dataclass
class TextSpan:
    role: str
    text: str
    char_start: int
    char_end: int
    token_indices: list

    def to_dict(self):
        return asdict(self)

# --- Keep EAGLE's helper functions unchanged ---
def _is_cjk(char):
    code = ord(char)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 
            0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF)

def _char_group(char):
    if char.isspace(): return "space"
    if _is_cjk(char): return "cjk"
    category = unicodedata.category(char)
    if category[0] in {"L", "N"} or char == "_": return "word"
    return "punct"

def split_readable_spans(text):
    spans = []
    index = 0
    while index < len(text):
        group = _char_group(text[index])
        if group == "space":
            index += 1
            continue
        start = index
        index += 1
        if group in {"word", "cjk"}:
            while index < len(text) and _char_group(text[index]) == group:
                index += 1
        spans.append((text[start:index], start, index))
    return spans

def _trim_span(text, start, end):
    while start < end and text[start].isspace(): start += 1
    while end > start and text[end - 1].isspace(): end -= 1
    if start >= end: return None
    return text[start:end], start, end

def split_sentence_spans(text):
    spans = []
    sentence_start = None
    sentence_end_chars = set(".!?;。！？；")
    closing_chars = set("\"')]}”’》」』）】")
    index = 0
    while index < len(text):
        char = text[index]
        if sentence_start is None:
            if char.isspace():
                index += 1
                continue
            sentence_start = index
        if char == "\n":
            span = _trim_span(text, sentence_start, index)
            if span is not None: spans.append(span)
            sentence_start = None
        elif char in sentence_end_chars:
            end = index + 1
            while end < len(text) and text[end] in closing_chars: end += 1
            span = _trim_span(text, sentence_start, end)
            if span is not None: spans.append(span)
            sentence_start = None
            index = end - 1
        index += 1
    if sentence_start is not None:
        span = _trim_span(text, sentence_start, len(text))
        if span is not None: spans.append(span)
    return spans

def split_text_spans(text, granularity="readable"):
    if granularity == "readable": return split_readable_spans(text)
    if granularity == "sentence": return split_sentence_spans(text)
    if granularity == "message": return [_trim_span(text, 0, len(text))] if _trim_span(text, 0, len(text)) else []
    raise ValueError(f"Unsupported text span granularity: {granularity}")

def _role_content(message):
    content = message.get("content", "")
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)

def _locate_message_ranges(rendered_text, messages):
    ranges = []
    cursor = 0
    for message in messages:
        role = message.get("role", "")
        if role not in {"system", "user"}: continue
        content = _role_content(message)
        if not content: continue
        start = rendered_text.find(content, cursor)
        if start < 0: start = rendered_text.find(content)
        if start < 0:
            raise ValueError(f"Could not locate {role!r} content in the rendered chat template.")
        end = start + len(content)
        ranges.append((role, content, start, end))
        cursor = end
    return ranges


# --- MODIFIED FOR VLM ---

def _tokens_for_char_range(offsets, start, end):
    token_indices = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        # Skip tokens that have a (0,0) offset (these are usually image or special tokens)
        if token_start == 0 and token_end == 0:
            continue
        if token_end <= start or token_start >= end:
            continue
        token_indices.append(token_index)
    return token_indices


def _vlm_token_offsets(processor, rendered_text, pre_generated_input_ids):
    """
    Uses your existing input_ids to align offsets, skipping redundant processor calls.
    """
    # 1. Get the raw character offsets from the text tokenizer
    encoded = processor.tokenizer(
        rendered_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded["offset_mapping"]
    
    # 2. Align the offsets with the pre-generated input_ids
    # (Injects (0,0) for all the expanded visual patches)
    if len(pre_generated_input_ids) != len(offsets):
        aligned_offsets = []
        offset_idx = 0
        for token_id in pre_generated_input_ids:
            token_str = processor.tokenizer.decode([token_id])
            if token_id == processor.tokenizer.pad_token_id or "image" in token_str.lower():
                aligned_offsets.append((0, 0)) 
            else:
                if offset_idx < len(offsets):
                    aligned_offsets.append(offsets[offset_idx])
                    offset_idx += 1
                else:
                    aligned_offsets.append((0,0))
        offsets = aligned_offsets

    return offsets


# (Assume TextSpan, split_text_spans, and _tokens_for_char_range are kept from the previous code)

def build_text_span_masks_from_inputs(
    processor,
    input_ids: torch.Tensor | list[int],
    target_text: str,
    granularity: str = "readable"
) -> tuple[np.ndarray, list[TextSpan]]:
    """
    Builds EAGLE token masks directly from the generated input_ids tensor.
    Uses Cumulative Decoding to guarantee perfect token-to-character alignment.
    """
    # 1. Flatten tensor to a 1D list
    if isinstance(input_ids, torch.Tensor):
        input_ids_list = input_ids.squeeze().tolist()
    else:
        input_ids_list = input_ids

    # 2. Decode the full sequence to find the exact starting position of the target text
    decoded_text = processor.tokenizer.decode(input_ids_list, skip_special_tokens=False)
    target_start = decoded_text.find(target_text)
    
    if target_start < 0:
        raise ValueError(
            f"Could not find the target text:\n'{target_text}'\n"
            f"Inside the decoded inputs:\n'{decoded_text}'"
        )

    # 3. Build perfect offsets using Cumulative Decoding
    # This completely bypasses the bug-prone return_offsets_mapping
    aligned_offsets = []
    
    for i in range(len(input_ids_list)):
        token_id = input_ids_list[i]
        token_str_raw = processor.tokenizer.decode([token_id])
        
        # If the token is a pad or visual placeholder, assign it a null (0,0) offset
        if token_id == processor.tokenizer.pad_token_id or "image" in token_str_raw.lower():
            aligned_offsets.append((0, 0))
        else:
            # Measure exact character lengths before and after this token is added
            prefix_text = processor.tokenizer.decode(input_ids_list[:i], skip_special_tokens=False)
            current_text = processor.tokenizer.decode(input_ids_list[:i+1], skip_special_tokens=False)
            
            start_char = len(prefix_text)
            end_char = len(current_text)
            aligned_offsets.append((start_char, end_char))

    seq_len = len(input_ids_list)
    spans = []
    masks = []

    # 4. Split the target text into chunks and map them to the aligned token indices
    for span_text, local_start, local_end in split_text_spans(target_text, granularity):
        char_start = target_start + local_start
        char_end = target_start + local_end
        
        # _tokens_for_char_range safely ignores the (0,0) image offsets
        token_indices = _tokens_for_char_range(aligned_offsets, char_start, char_end)
        if not token_indices:
            continue

        mask = np.zeros(seq_len, dtype=np.float32)
        mask[token_indices] = 1.0
        masks.append(mask)
        spans.append(
            TextSpan(
                role="user", 
                text=span_text,
                char_start=char_start,
                char_end=char_end,
                token_indices=token_indices,
            )
        )

    # 5. Stack into the final EAGLE mask array
    mask_array = np.stack(masks, axis=0) if masks else np.zeros((0, seq_len), dtype=np.float32)

    return mask_array, spans




def SubRegionDivision(image, mode="slico", region_size=30):
    element_sets_V = []
    
    if mode == "slico":
        # SLIC / SLICO (SLICO est activé par défaut via l'algorithme de base dans ximgproc)
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
        slic.iterate(20)
        labels = slic.getLabels()
        number_superpixels = slic.getNumberOfSuperpixels()

    elif mode == "seeds":
        # SEEDS : Correction des dimensions (width, height, channels)
        h, w, c = image.shape
        seeds = cv2.ximgproc.createSuperpixelSEEDS(w, h, c, num_superpixels=50, num_levels=3)
        seeds.iterate(image, 10)
        labels = seeds.getLabels()
        number_superpixels = seeds.getNumberOfSuperpixels()
    else:
        raise ValueError("Mode non supporté. Utilisez 'slico' ou 'seeds'.")

    # Extraction des masques
    for i in range(number_superpixels):
        # Correction : Utilisation de 'labels' (qui contient soit label_slic soit label_seeds)
        # Ajout de .astype(np.uint8) pour économiser de la mémoire par rapport à int
        img_copp = (labels == i)[:, :, np.newaxis].astype(np.uint8)
        element_sets_V.append(img_copp)
        
    return element_sets_V
