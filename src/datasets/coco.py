import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class COCOGroundingDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to image folder (e.g., 'val2017')
        """
        self.data_path = data_path
        self.coco_split = "val2017"
        instances_file = os.path.join(
            self.data_path, "annotations", f"instances_{self.coco_split}.json"
        )
        captions_file = os.path.join(
            self.data_path, "annotations", f"captions_{self.coco_split}.json"
        )

        self.coco_instances = COCO(instances_file)
        self.coco_captions = COCO(captions_file)

        # Filter: Only keep images that have annotations AND captions
        # (Usually identical sets, but good for safety)
        self.ids = sorted(self.coco_instances.imgs.keys())

        # Cache category mapping for the Instances
        cats = self.coco_instances.loadCats(self.coco_instances.getCatIds())
        self.id2name = {cat["id"]: cat["name"] for cat in cats}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # --- 1. Load Image ---
        img_info = self.coco_instances.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_path, self.coco_split, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # --- 2. Load Captions ---
        # Get caption annotation IDs for this image
        ann_ids_cap = self.coco_captions.getAnnIds(imgIds=img_id)
        anns_cap = self.coco_captions.loadAnns(ann_ids_cap)

        # COCO has 5 captions per image. Pick one randomly (training) or the first (eval).
        # For a benchmark, it is better to return ALL of them or a specific one.
        # Let's pick the first one for consistency.
        captions_list = [ann["caption"] for ann in anns_cap]

        # --- Load Instance Masks (Ground Truth) ---
        ann_ids_inst = self.coco_instances.getAnnIds(imgIds=img_id)
        anns_inst = self.coco_instances.loadAnns(ann_ids_inst)

        category_masks = {}
        for ann in anns_inst:
            cat_name = self.id2name[ann["category_id"]]

            # Create mask for this specific instance
            instance_mask = self.coco_instances.annToMask(ann)

            # Union logic: If we already have a mask for "dog", add this new dog to it
            if cat_name in category_masks:
                category_masks[cat_name] = np.maximum(
                    category_masks[cat_name], instance_mask
                )
            else:
                category_masks[cat_name] = instance_mask

        category_masks_tensor = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in category_masks.items()
        }

        return {
            "image": image,
            "captions": captions_list,
            "category_masks": category_masks_tensor,  # Dict[str, Tensor]
            "image_id": img_id,
            "image_path": img_path,
        }
