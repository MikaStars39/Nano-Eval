import json
from io import TextIOWrapper
from tqdm import tqdm
from ._base import load_hf_dataset, apply_box_prompt, apply_lettered_options, digit_to_letter


def load_mmlu(cache_dir: str, k: int, f_out: TextIOWrapper):
    dataset_name = "mmlu"
    dataset = load_hf_dataset(dataset_name, cache_dir)
    
    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {dataset_name}")):
        question = row["question"]

        digit_answer = row["answer"]
        answer = digit_to_letter(digit_answer)

        options = row["choices"]
        lettered_prompt = apply_lettered_options(question, options)
        prompt = apply_box_prompt(lettered_prompt)

        for sample_idx in range(k):
            record = {
                "id": f"{dataset_name}_{idx}_{sample_idx}",
                "prompt": prompt,
                "need_llm_extract": False,
                "label": answer,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
