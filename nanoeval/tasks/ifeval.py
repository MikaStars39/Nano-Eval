from typing import Any


import json
from io import TextIOWrapper
from tqdm import tqdm
from ._base import load_hf_dataset


def load_ifeval(cache_dir: str, k: int, f_out: TextIOWrapper):
    dataset_name = "ifeval"
    dataset = load_hf_dataset(dataset_name, cache_dir)
    
    for idx, row in enumerate[Any](tqdm(dataset, desc=f"Loading {dataset_name}")):
        for sample_idx in range(k):
            record = {
                "id": f"{dataset_name}_{idx}_{sample_idx}",
                "need_llm_extract": False,
                **row
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
