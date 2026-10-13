import pandas as pd
from torch.utils.data import Dataset

from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset


class MMVPDataset(Dataset):
    def __init__(self, hf_path="MMVP/MMVP", split="train", transform=None):
        """
        Args:
            hf_path (str): The Hugging Face dataset path.
            split (str): The dataset split to load.
            transform (callable, optional): Optional transform for the PIL image.
        """
        # 1. Load the actual images/data from HF
        self.hf_dataset = load_dataset(hf_path, split=split)
        self.transform = transform

        # 2. Load the metadata CSV directly from the HF repository
        # This fills the "missing questions" gap in the HF dataset object
        csv_url = f"https://huggingface.co/datasets/{hf_path}/raw/main/Questions.csv"
        self.metadata = pd.read_csv(csv_url)

    def __len__(self):
        if isinstance(self.hf_dataset, (HFDataset, DatasetDict)):
            return len(self.hf_dataset)
        else:
            raise TypeError("Dataset is streaming (IterableDataset) and has no length.")

    def __getitem__(self, index):
        # HF dataset items contain {'image': <PIL>, 'label': ...}
        item = self.hf_dataset[index]
        
        # Metadata contains ['Index', 'Question', 'Options', 'Correct Answer']
        # We assume the HF dataset order matches the CSV Index
        meta = self.metadata.iloc[index]

        image = item['image'] #.convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # Formatting the prompt for a VLM
        question = meta['Question']
        options = meta['Options']
        full_prompt = f"{question}\n{options}"
        
        return {
            "image": image,
            "question": full_prompt,
            "label": str(meta['Correct Answer']).strip(),
            "index": int(meta['Index']),
            "metadata": {
                "raw_question": question,
                "options": options
            }
        }

