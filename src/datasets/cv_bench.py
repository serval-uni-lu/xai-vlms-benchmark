from torch.utils.data import Dataset

from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset


class CVBenchDataset(Dataset):
    def __init__(self, hf_path="nyu-visionx/CV-Bench", split="test"):
        """
        Args:
            hf_path (str): The Hugging Face dataset path.
            split (str): The dataset split to load.
        """
        # 1. Load the actual images/data from HF
        self.hf_dataset = load_dataset(hf_path, split=split)

    def __len__(self):
        if isinstance(self.hf_dataset, (HFDataset, DatasetDict)):
            return len(self.hf_dataset)
        else:
            raise TypeError("Dataset is streaming (IterableDataset) and has no length.")

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        

        image = item['image'] #.convert("RGB")
        idx = item["idx"]
        question = item['prompt']
        label = item["answer"]
        category = item["task"]
        
        return {
            "image": image,
            "question": question,
            "label": label,
            "index": idx,
            "category": category,
        }

