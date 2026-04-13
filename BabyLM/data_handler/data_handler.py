from datasets import load_dataset
from BabyLM.config import DATASET_PATH

def load_data(download_locally=True):
    dataset_name = "BabyLM-community/BabyLM-2026-Strict-Small"
    if download_locally:
        dataset = load_dataset(
            dataset_name,
            cache_dir=DATASET_PATH
        )
    else:
        dataset = load_dataset(dataset_name, streaming=True)

    return dataset
