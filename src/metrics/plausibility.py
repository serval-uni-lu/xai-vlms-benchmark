import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Assuming BaseMetric and safe_flatten_to_list are defined in your utils/base classes
from src.metrics.base import BaseMetric

# from src.metrics.plausibility_utils import safe_flatten_to_list


# Helper to safely convert arrays/tensors of shape (S, B) to flat Python lists
def safe_flatten_to_list(array_like):
    if hasattr(array_like, "flatten"):
        return array_like.flatten().tolist()
    elif hasattr(array_like, "tolist"):
        flat_list = []
        for item in array_like.tolist():
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        return flat_list
    return list(array_like)


class PlausibilityMetric(BaseMetric):
    def __init__(
        self,
        ontology_mapper,
        category_dict: dict,
    ):
        super().__init__("Plausibility")
        self.mapper = ontology_mapper
        self.category_dict = category_dict

    def get_valid_targets(self, words: list[str], tokens_id_groups: list[list[int]], masks: dict):
        """
        Pre-filters the caption to find valid nouns and their token IDs.
        Call this BEFORE the explainer to save VRAM and compute time.
        """
        valid_words = []
        target_indices = []
        local_row_counter = 0

        for i, word in enumerate(words):
            cat_id = self.mapper.map_word(word)
            if cat_id is None:
                continue

            cat_name = self.category_dict.get(cat_id)
            if not cat_name or cat_name not in masks:
                continue  # Hallucination or object not in ground truth

            absolute_ids = tokens_id_groups[i]
            target_indices.extend(absolute_ids)

            local_indices = list(range(local_row_counter, local_row_counter + len(absolute_ids)))
            local_row_counter += len(absolute_ids)

            valid_words.append({"word": word, "cat_name": cat_name, "local_indices": local_indices})

        return valid_words, target_indices

    def compute(
        self,
        wrapper,  # Type: BaseVLMWrapper
        sample: dict[str, Any],
        xai_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Executes Obj-IoU, Point Game, and Energy Game metrics.
        """
        results = {}
        start_time = time.perf_counter()

        img_attrs = xai_result.get("pixel_attribution")
        valid_words = xai_result.get("valid_words")
        inputs = xai_result.get("inputs", {})

        masks = sample.get("category_masks", {})

        # Handle edge case: Model hallucinated or explainer failed
        if not valid_words or img_attrs is None or len(img_attrs) == 0:
            raw_metrics = {"obj_iou": 0.0, "point_game": 0.0, "energy_score": 0.0}
            results["time_plausibility"] = time.perf_counter() - start_time
            results.update(self._format_results("plaus", raw_metrics))
            return results

        # 1. Reshape Heatmaps to 2D Spatial Grids
        image_grid_thw = inputs.get("image_grid_thw", None)
        img_attrs_2d = self._reshape_to_2d_grid(img_attrs, image_grid_thw)

        obj_ious, point_games, energy_scores = [], [], []

        # 2. Iterate through objects and compute scores
        for item in valid_words:
            cat_name = item["cat_name"]
            local_idx = item["local_indices"]

            gt_mask = masks[cat_name].numpy()
            gt_mask_binary = (gt_mask > 0).astype(np.uint8)
            h_gt, w_gt = gt_mask_binary.shape

            # Pool across tokens for this specific word
            word_heatmaps = img_attrs_2d[local_idx]
            img_attr_pooled = word_heatmaps.max(dim=0)[0]

            # Interpolate safely
            heatmap_raw = img_attr_pooled.unsqueeze(0).unsqueeze(0).float()
            heatmap_resized = (
                F.interpolate(heatmap_raw, size=(h_gt, w_gt), mode="bilinear", align_corners=False)
                .squeeze()
                .cpu()
                .numpy()
            )

            obj_ious.append(self._obj_iou(heatmap_resized, gt_mask_binary))
            point_games.append(self._point_game(heatmap_resized, gt_mask_binary))
            energy_scores.append(self._energy_score(heatmap_resized, gt_mask_binary))

        # 3. Aggregate
        raw_metrics = {
            "obj_iou": sum(obj_ious) / len(obj_ious) if obj_ious else 0.0,
            "point_game": sum(point_games) / len(point_games) if point_games else 0.0,
            "energy_score": sum(energy_scores) / len(energy_scores) if energy_scores else 0.0,
        }

        results["time_plausibility"] = time.perf_counter() - start_time
        results.update(self._format_results("plaus", raw_metrics))

        return results

    def _format_results(self, prefix: str, raw_metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Formats floats and lists for safe W&B logging with modality prefixes.
        """
        formatted = {}
        for key, val in raw_metrics.items():
            new_key = f"{prefix}_{key}"

            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()

            if isinstance(val, np.ndarray) and val.size == 1:
                formatted[new_key] = float(val.item())
            elif isinstance(val, np.ndarray):
                formatted[new_key] = safe_flatten_to_list(val)
            elif isinstance(val, (float, int)):
                formatted[new_key] = val
            else:
                formatted[new_key] = val

        return formatted

    # -------------------------------------------------------------------------
    # INTERNAL MATH AND SHAPE HELPERS
    # -------------------------------------------------------------------------

    def _reshape_to_2d_grid(self, img_attrs: torch.Tensor, image_grid_thw=None):
        num_tokens = img_attrs.shape[0]
        if img_attrs.dim() == 2:
            num_patches = img_attrs.shape[1]
            if image_grid_thw is not None:
                grid_h = int(image_grid_thw[0, 1].item())
                grid_w = int(image_grid_thw[0, 2].item())
            else:
                grid_h = grid_w = int(np.sqrt(num_patches))
            return img_attrs.view(num_tokens, grid_h, grid_w)
        elif img_attrs.dim() == 3:
            return img_attrs
        elif img_attrs.dim() == 4:
            return img_attrs[:, -1, :, :]
        else:
            raise ValueError(f"Unexpected img_attr shape: {img_attrs.shape}")

    def _obj_iou(self, heatmap_resized: np.ndarray, gt_mask: np.ndarray) -> float:
        h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
        if h_max > h_min:
            heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min)
            heatmap_8bit = (heatmap_norm * 255).astype(np.uint8)
        else:
            heatmap_8bit = np.zeros_like(heatmap_resized, dtype=np.uint8)

        _, pred_binary = cv2.threshold(heatmap_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        intersection = np.logical_and(gt_mask > 0, pred_binary > 0).sum()
        union = np.logical_or(gt_mask > 0, pred_binary > 0).sum()
        return 0.0 if union == 0 else float(intersection) / float(union)

    def _point_game(self, heatmap_resized: np.ndarray, gt_mask: np.ndarray) -> int:
        if heatmap_resized.max() == heatmap_resized.min():
            return 0
        mask_bbox = heatmap_resized * gt_mask
        if (
            mask_bbox.max() == heatmap_resized.max()
            and heatmap_resized.max() > 0
            or gt_mask[heatmap_resized == heatmap_resized.max()].any()
        ):
            return 1
        return 0

    def _energy_score(self, heatmap_resized: np.ndarray, gt_mask: np.ndarray) -> float:
        pos_saliency = np.maximum(heatmap_resized, 0)
        mask_bbox = pos_saliency * gt_mask
        energy_whole = pos_saliency.sum()
        return 0.0 if energy_whole == 0 else float(mask_bbox.sum() / energy_whole)
