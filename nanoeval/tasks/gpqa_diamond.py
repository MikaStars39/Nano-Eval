from datasets import load_dataset
from tqdm import tqdm
import json
from io import TextIOWrapper

def load_gpqa_diamond(
    cache_dir: str,
    k: int,
    f_out: TextIOWrapper,
):
    dataset_name="gpqa_diamond"
    dataset = load_dataset(cache_dir, split="test")
    
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
        
        question = row["question"]
        answer = row["answer"]

        for sample_idx in range(k):
            
            unique_id = f"{dataset_name}_{idx}_{sample_idx}"
            record = {
                "id": unique_id,
                "prompt": question + "\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                "need_llm_extract": False,
                "label": answer,
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
