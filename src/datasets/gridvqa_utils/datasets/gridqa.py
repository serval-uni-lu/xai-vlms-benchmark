"""
GridQA Dataset Loader

This module loads data points from the GridQA dataset, which contains
visual question answering samples on grid-based images with ground truth
masks for explainability analysis.

Dataset structure:
    grid_vqa_data/
        val/
            qa_test.jsonl           # Question-answer pairs
            grounding_test.jsonl    # Grounding annotations
            depth1/                 # Question depth
                form0/              # Template form
                    A/              # Category (A, CMP, CO, M, SO)
                        g5_d03/     # Grid size / density
                            000000.npz
                            ...

Each .npz file contains:
    - id: int, global sample ID
    - image: np.ndarray (H, W, 3), RGB image
    - question: str, natural language question
    - answer: int, numeric answer
    - objects: list of object descriptions
    - cell_mask: np.ndarray, cell-level ground truth mask
    - pixel_mask: np.ndarray, pixel-level ground truth mask
    - text_mask: np.ndarray, text token relevance mask
    - pos_mask: np.ndarray, positional mask
"""

import os
import sys
import glob
import random
import numpy as np
from PIL import Image

sys.path.insert(1, os.getcwd())


# Default data root (can be overridden via the GRIDVQA_DATA_ROOT env var or the
# data_root constructor argument). The .npz analysis data is hosted on the
# Hugging Face dataset repo: Aikyam-Lab/gridvqa-dataset
GRIDVQA_DATA_ROOT = "./data/gridvqa"
DEFAULT_DATA_ROOT = os.environ.get("GRIDVQA_DATA_ROOT", "density30_pure")
# DEFAULT_DATA_ROOT = os.environ.get("GRIDVQA_DATA_ROOT", "d07_pure_data")


class GridQADataset:
    """
    GridQA Dataset loader for visual question answering with ground truth masks.
    
    Parameters
    ----------
    split : str
        Dataset split, e.g., "val". Currently only "val" is supported.
    data_root : str, optional
        Root directory of the GridQA dataset.
    depth : str or list, optional
        Question depth(s) to include, e.g., "depth1", "depth2", "depth3" or ["depth1", "depth2"].
        Default is None (all depths).
    form : str or list, optional
        Form(s) to include, e.g., "form0", "form1" or ["form0", "form1"].
        Default is None (all forms).
    category : str or list, optional
        Category/categories to include, e.g., "CO", "A", "CMP", "M", "SO".
        Default is None (all categories).
    grid_config : str or list, optional
        Grid configuration(s) to include, e.g., "g5_d03".
        Default is None (all configurations).
    """

    def __init__(
        self,
        split="val",
        data_root=DEFAULT_DATA_ROOT,
        depth=None,
        form=None,
        category=None,
        grid_config=None,
    ):
        self.split = split
        self.data_root = data_root
        self.split_dir = os.path.join(data_root, split)
        
        # Filters
        self.depth_filter = self._to_list(depth)
        self.form_filter = self._to_list(form)
        self.category_filter = self._to_list(category)
        self.grid_config_filter = self._to_list(grid_config)
        
        # Discover and index all .npz files
        self.samples = self._discover_samples()
        
        # Answer mapping: for counting questions, answers are typically 0-25
        # Adjust based on your dataset's answer range
        self.answermapping = [str(i) for i in range(26)]
        print("GridVQA Dataset Path:", self.split_dir)

    def _to_list(self, val):
        """Convert a single value or None to a list for filtering."""
        if val is None:
            return None
        if isinstance(val, str):
            return [val]
        return list(val)

    def _discover_samples(self):
        """
        Discover all .npz sample files in the dataset directory.
        
        Returns a list of dicts with metadata:
            {
                'path': str,       # Full path to .npz file
                'depth': str,      # e.g., 'depth1'
                'form': str,       # e.g., 'form0'
                'category': str,   # e.g., 'CO'
                'grid_config': str # e.g., 'g5_d03'
            }
        """
        samples = []
        
        # Pattern: split_dir/depth*/form*/category/grid_config/*.npz
        pattern = os.path.join(self.split_dir, "depth*", "form*", "*", "*", "*.npz")
        npz_files = glob.glob(pattern)
        
        for npz_path in sorted(npz_files):
            # Parse path to extract metadata
            rel_path = os.path.relpath(npz_path, self.split_dir)
            parts = rel_path.split(os.sep)
            
            if len(parts) >= 5:
                depth, form, category, grid_config, filename = parts[:5]
                
                # Apply filters
                if self.depth_filter and depth not in self.depth_filter:
                    continue
                if self.form_filter and form not in self.form_filter:
                    continue
                if self.category_filter and category not in self.category_filter:
                    continue
                if self.grid_config_filter and grid_config not in self.grid_config_filter:
                    continue
                
                samples.append({
                    'path': npz_path,
                    'depth': depth,
                    'form': form,
                    'category': category,
                    'grid_config': grid_config,
                })
        
        return samples

    def getdata(self, idx):
        """
        Get a single data point by index.
        
        Parameters
        ----------
        idx : int
            Index of the sample.
        
        Returns
        -------
        tuple
            (image_path_or_array, question, answer, label)
            - image: np.ndarray (H, W, 3), RGB image
            - question: str
            - answer: int or str
            - label: int (index into answermapping)
        """
        sample_meta = self.samples[idx]
        npz_path = sample_meta['path']
        
        data = np.load(npz_path, allow_pickle=True)
        
        image = data['image']
        question = str(data['question'])
        answer = int(data['answer'])
        
        # Label is the index in answermapping
        try:
            label = self.answermapping.index(str(answer))
        except ValueError:
            label = None
        
        return image, question, answer, label

    def getdata_full(self, idx):
        """
        Get a single data point with all attributes including masks.
        
        Parameters
        ----------
        idx : int
            Index of the sample.
        
        Returns
        -------
        dict
            Dictionary containing all attributes from the .npz file plus metadata.
        """
        sample_meta = self.samples[idx]
        npz_path = sample_meta['path']
        
        data = np.load(npz_path, allow_pickle=True)
        
        result = {
            'id': int(data['id']),
            'image': data['image'],
            'question': str(data['question']),
            'answer': data['answer'],
            'objects': data['objects'],
            'cell_mask': data['cell_mask'],
            'pixel_mask': data['pixel_mask'],
            'text_mask': data['text_mask'],
            'pos_mask': data['pos_mask'],
            # Metadata from path
            'depth': sample_meta['depth'],
            'form': sample_meta['form'],
            'category': sample_meta['category'],
            'grid_config': sample_meta['grid_config'],
            'path': npz_path,
        }
        
        return result

    def length(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length()

    def classnames(self):
        """Return the answer mapping (class names)."""
        return self.answermapping

    def sample(self, num, noNone=True):
        """
        Randomly sample `num` data points.
        
        Parameters
        ----------
        num : int
            Number of samples to return.
        noNone : bool
            If True, skip samples where label is None.
        
        Returns
        -------
        list
            List of (image, question, answer, label) tuples.
        """
        sampled = []
        indices = list(range(self.length()))
        random.shuffle(indices)
        
        idx = 0
        while len(sampled) < num and idx < len(indices):
            data = self.getdata(indices[idx])
            if data[-1] is not None or not noNone:
                sampled.append(data)
            idx += 1
        
        return sampled

    def sample_full(self, num):
        """
        Randomly sample `num` data points with full attributes.
        
        Parameters
        ----------
        num : int
            Number of samples to return.
        
        Returns
        -------
        list
            List of dicts with all attributes.
        """
        sampled = []
        indices = list(range(self.length()))
        random.shuffle(indices)
        
        for i in range(min(num, len(indices))):
            sampled.append(self.getdata_full(indices[i]))
        
        return sampled

    def getseqdata(self, start, end):
        """
        Get a sequential range of data points.
        
        Parameters
        ----------
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).
        
        Returns
        -------
        list
            List of (image, question, answer, label) tuples.
        """
        return [self.getdata(i) for i in range(start, min(end, self.length()))]

    def get_categories(self):
        """Return unique categories in the loaded dataset."""
        return sorted(set(s['category'] for s in self.samples))

    def get_depths(self):
        """Return unique depths in the loaded dataset."""
        return sorted(set(s['depth'] for s in self.samples))

    def get_forms(self):
        """Return unique forms in the loaded dataset."""
        return sorted(set(s['form'] for s in self.samples))

    def get_grid_configs(self):
        """Return unique grid configurations in the loaded dataset."""
        return sorted(set(s['grid_config'] for s in self.samples))

    def filter_by(self, depth=None, form=None, category=None, grid_config=None):
        """
        Return indices of samples matching the given filters.
        
        Parameters
        ----------
        depth : str or list, optional
        form : str or list, optional
        category : str or list, optional
        grid_config : str or list, optional
        
        Returns
        -------
        list
            List of indices matching the filters.
        """
        depth_list = self._to_list(depth)
        form_list = self._to_list(form)
        category_list = self._to_list(category)
        grid_config_list = self._to_list(grid_config)
        
        indices = []
        for i, s in enumerate(self.samples):
            if depth_list and s['depth'] not in depth_list:
                continue
            if form_list and s['form'] not in form_list:
                continue
            if category_list and s['category'] not in category_list:
                continue
            if grid_config_list and s['grid_config'] not in grid_config_list:
                continue
            indices.append(i)
        
        return indices


def download_data():
    """Point users to the Hugging Face dataset repo for the analysis .npz data."""
    print("Download the GridVQA analysis data from the Hugging Face Hub:")
    print("  https://huggingface.co/datasets/Aikyam-Lab/gridvqa-dataset")
    print("e.g.:")
    print("  huggingface-cli download Aikyam-Lab/gridvqa-dataset --repo-type dataset \\")
    print(f"      --local-dir {DEFAULT_DATA_ROOT}")
    print(f"Then point data_root / GRIDVQA_DATA_ROOT at it (default: {DEFAULT_DATA_ROOT}/).")
    print("Expected structure: <data_root>/val/depth*/form*/category/grid_config/*.npz")


# Convenience function to create dataset
def get_gridqa_dataset(split="val", **kwargs):
    """
    Factory function to create a GridQADataset instance.
    
    Parameters
    ----------
    split : str
        Dataset split.
    **kwargs
        Additional arguments passed to GridQADataset constructor.
    
    Returns
    -------
    GridQADataset
    """
    return GridQADataset(split=split, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing GridQADataset...")
    
    dataset = GridQADataset(split="val")
    print(f"Total samples: {dataset.length()}")
    print(f"Depths: {dataset.get_depths()}")
    print(f"Forms: {dataset.get_forms()}")
    print(f"Categories: {dataset.get_categories()}")
    print(f"Grid configs: {dataset.get_grid_configs()}")
    
    if dataset.length() > 0:
        print("\nSample data point:")
        img, question, answer, label = dataset.getdata(0)
        print(f"  Image shape: {img.shape}")
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")
        print(f"  Label: {label}")
        
        print("\nFull data point:")
        full = dataset.getdata_full(0)
        for k, v in full.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")
