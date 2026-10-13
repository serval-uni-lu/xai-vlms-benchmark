from torch.utils.data import Dataset

from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset




class MMStarDataset(Dataset):
    def __init__(self, hf_path="Lin-Chen/MMStar", split="val"):
        """
        Args:
            hf_path (str): The Hugging Face dataset path.
            split (str): The dataset split to load.
        """
        # 1. Load the actual images/data from HF
        self.hf_dataset = load_dataset(hf_path, split=split, streaming=False)

    def __len__(self):
        if isinstance(self.hf_dataset, (HFDataset, DatasetDict)):
            return len(self.hf_dataset)
        else:
            raise TypeError("Dataset is streaming (IterableDataset) and has no length.")
        

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        

        image = item['image'] #.convert("RGB")
        idx = item["index"]
        question = item['question']
        meta_info = item['meta_info']
        label = item["answer"]
        category = item["category"]
        
        return {
            "image": image,
            "question": question,
            "label": label,
            "index": idx,
            "metadata": meta_info,
            "category": category,
        }

