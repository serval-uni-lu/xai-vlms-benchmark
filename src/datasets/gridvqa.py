import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

# Import the base loader provided by GridVQA
# (Assuming the original class is saved in gridvqa_loader.py)
from src.datasets.gridvqa_utils.datasets.gridqa import GridQADataset as BaseGridQADataset


class GridQAMatchedDataset(Dataset):
    def __init__(self, processor, data_root=None, split="val", category=None):
        """
        Pass the Qwen2-VL processor into the dataset so we can align 
        the masks to the exact tensor shapes the model expects.
        """
        self.processor = processor
        kwargs = {"split": split}
        if data_root: kwargs["data_root"] = data_root
        if category: kwargs["category"] = category

        self.grid_dataset = BaseGridQADataset(**kwargs)

    def __len__(self) -> int:
        return len(self.grid_dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.grid_dataset.getdata_full(index)

        # 1. Image Formatting
        raw_img = item["image"]
        image = Image.fromarray(raw_img.astype(np.uint8)).convert("RGB") if isinstance(raw_img, np.ndarray) else raw_img

        # 2. Text Formatting
        question = item["question"]
        label = str(item["answer"])

        # ==========================================
        # 3. ALIGN PIXEL MASK TO SPATIAL GRID (H, W)
        # ==========================================
        # Run the image through the processor to get the exact output shape
        image_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        
        # Determine the spatial dimensions (H, W) depending on the processor type
        if pixel_values.ndim == 4:
            # Standard LLaVA / CLIP format: (Batch, Channels, H, W)
            grid_h, grid_w = pixel_values.shape[2], pixel_values.shape[3]
        elif "image_grid_thw" in image_inputs:
            # Fallback if you still pass it through Qwen's processor
            grid_thw = image_inputs["image_grid_thw"][0]
            grid_h, grid_w = grid_thw[1].item(), grid_thw[2].item()
        else:
            # Absolute fallback: use the original image size
            grid_w, grid_h = image.size

        # Convert raw NumPy mask to PyTorch (1, 1, raw_H, raw_W)
        raw_pixel_mask = torch.from_numpy(item["pixel_mask"]).float().unsqueeze(0).unsqueeze(0)
        
        # Resize to the processor's grid and SQUEEZE to (H, W) instead of flattening!
        model_pixel_mask = F.interpolate(raw_pixel_mask, size=(grid_h, grid_w), mode="nearest")
        model_pixel_mask = model_pixel_mask.squeeze().bool() # Shape: (H, W)

        # ==========================================
        # 4. ALIGN TEXT MASK TO TOKENIZER (seq_len,)
        # ==========================================
        text_inputs = self.processor(text=question, return_tensors="pt")
        input_ids = text_inputs["input_ids"][0] # Shape: (seq_len,)
        
        model_text_mask = self._build_token_mask(input_ids, item["objects"])

        # 5. Package perfectly model tensors
        meta_info = {
            "pixel_mask": item["pixel_mask"],               # Raw Numpy
            "model_pixel_mask": model_pixel_mask,       # PyTorch (H, W)
            "text_mask": item["text_mask"],                 # Raw Numpy
            "model_text_mask": model_text_mask,         # PyTorch (seq_len,)
            "raw_objects": item["objects"],
            "grid_config": item["grid_config"],
        }

        return {
            "image": image,
            "question": question,
            "label": label,
            "index": item["id"],
            "metadata": meta_info,
            "category": item["category"],
        }



    def _build_token_mask(self, input_ids: torch.Tensor, objects: list | np.ndarray) -> torch.Tensor:
        """Finds the character spans of objects and activates overlapping BPE tokens."""
        device = input_ids.device
        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        ids_list = input_ids.tolist()
        
        current_str = ""
        token_char_spans = []
        for tid in ids_list:
            token_str = self.processor.tokenizer.decode([tid], skip_special_tokens=False)
            start_char = len(current_str)
            current_str += token_str
            token_char_spans.append((start_char, len(current_str), token_str))
            
        prompt_lower = current_str.lower()
        search_terms = set()
        
        if isinstance(objects, np.ndarray):
            objects = objects.tolist()
            
        for obj in objects:
            if isinstance(obj, dict):
                if 'name' in obj: search_terms.add(str(obj['name']).lower().strip())
                else:
                    c, s = str(obj.get('color', '')).lower().strip(), str(obj.get('shape', '')).lower().strip()
                    if c and s: search_terms.add(f"{c} {s}")
                    if c: search_terms.add(c)
                    if s: search_terms.add(s)
            else:
                search_terms.add(str(obj).lower().strip())
                
        for term in search_terms:
            if not term: continue
            start_idx = 0
            while True:
                target_start = prompt_lower.find(term, start_idx)
                if target_start == -1: break
                target_end = target_start + len(term)
                
                for i, (start, end, text) in enumerate(token_char_spans):
                    if max(start, target_start) < min(end, target_end) and text.strip():
                        mask[i] = True
                start_idx = target_start + 1
                
        return mask
