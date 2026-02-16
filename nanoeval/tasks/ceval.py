import json
from io import TextIOWrapper
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from ._base import load_ceval_all_subdomains,apply_lettered_options, apply_box_prompt


def load_ceval(cache_dir: str, k: int, f_out: TextIOWrapper):
    datasets_iter = load_ceval_all_subdomains(cache_dir)
    dataset_name = "ceval"
    # info = DATASETS[dataset_name]
    # hf_name = info["hf_name"]
    # split = info["split"]

    
    for subset, dataset in datasets_iter:
        for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}/{subset}")):
            question = row["question"]
            answer = str(row["answer"])
            options = [f"{letter}. {row[letter]}" for letter in ["A", "B", "C", "D"] if letter in row]
            lettered_prompt = apply_lettered_options(question,options)
            prompt= apply_box_prompt(lettered_prompt)    

            
            for sample_idx in range(k):
                record = {
                    "id": f"{dataset_name}_{subset}_{idx}_{sample_idx}",
                    "prompt": prompt,
                    "need_llm_extract": False,
                    "label": answer,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
