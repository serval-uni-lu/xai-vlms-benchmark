from torch.utils.data import DataLoader

from src.datasets import (
    COCOGroundingDataset,
    CVBenchDataset,
    MMStarDataset,
    MMVPDataset,
    POPEGroundingDataset,
    POPEOracleDataset,
)


def get_dataloader(dataset_config: dict):
    """
    Dynamically loads a dataset and wraps it in a DataLoader
    based on a YAML config dictionary.
    """
    # Copy the config so we can pop items safely without modifying the original
    config_copy = dataset_config.copy()

    # Extract the routing variables
    dataset_name = config_copy.pop("name")
    batch_size = config_copy.pop("batch_size", 1)

    print(f"[*] Loading Dataset: {dataset_name}")

    if dataset_name == "repope":
        dataset = POPEGroundingDataset(**config_copy)
    elif dataset_name == "repope_oracle":
        dataset = POPEOracleDataset(**config_copy)
    elif dataset_name == "coco":
        dataset = COCOGroundingDataset(**config_copy)
    elif dataset_name == "mmvp":
        dataset = MMVPDataset(**config_copy)
    elif dataset_name == "mmstar":
        dataset = MMStarDataset(**config_copy)
    elif dataset_name == "cv_bench":
        dataset = CVBenchDataset(**config_copy)
    else:
        raise ValueError(f"[!] Unknown dataset name in config: {dataset_name}")

    # VLM datasets usually return complex dicts/lists, so we use a simple collate function
    def collate_fn_custom(batch):
        return batch[0] if batch_size == 1 else batch

    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_custom)
    return dl
