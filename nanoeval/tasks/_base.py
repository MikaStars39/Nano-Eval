from datasets import load_dataset, get_dataset_config_names


DATASETS = {
    "aime2024": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "eval_type": "math",
    },
    "aime2025": {
        "hf_name": "yentinglin/aime_2025",
        "split": "train",
        "eval_type": "math",
    },
    "amc2023": {
        "hf_name": "zwhe99/amc23",
        "split": "test",
        "eval_type": "math",
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "eval_type": "math",
    },
    "minerva": {
        "hf_name": "math-ai/minervamath",
        "split": "test",
        "eval_type": "math",
    },
    "hmmt2025": {
        "hf_name": "FlagEval/HMMT_2025",
        "split": "train",
        "eval_type": "math",
    },
    "gpqa_diamond": {
        "hf_name": "fingertap/GPQA-Diamond",
        "split": "test",
        "eval_type": "qa",
    },
    "mmlu_pro": {
        "hf_name": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "eval_type": "qa",
    },
    "mmlu": {
        "hf_name": "cais/mmlu",
        "split": "test",
        "eval_type": "qa",
        "config": "all",
    },
    "ceval": {
        "hf_name": "ceval/ceval-exam",
        "split": "test",
        "eval_type": "qa",
    },
    "ifeval": {
        "hf_name": "google/IFEval",
        "split": "train",
        "eval_type": "instruction",
    },
}


def load_hf_dataset(dataset_name: str, cache_dir: str = None):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    info = DATASETS[dataset_name]
    hf_name = info["hf_name"]
    split = info["split"]
    config = info.get("config")
    
    if config:
        return load_dataset(hf_name, config, split=split, cache_dir=cache_dir)
    return load_dataset(hf_name, split=split, cache_dir=cache_dir)


# apply this, except for the instruction 
def apply_box_prompt(question: str) -> str:
    return f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# apply this for mmlu and mmlu_pro
def apply_lettered_options(question: str, options: list[str]) -> str:
    lettered_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    return f"{question}\n\nOptions:\n" + "\n".join(lettered_options)

def digit_to_letter(digit: int) -> str:
    return chr(65 + digit)


def load_ceval_all_subdomains(cache_dir: str = None):
    dataset_name = "ceval"

    info = DATASETS[dataset_name]
    hf_name = info["hf_name"]
    split = info["split"]

    datasets_iter=[]
    subset_names = get_dataset_config_names(hf_name,cache_dir=cache_dir)
    for subset in subset_names:
        print("Loading subset of ceval: ", subset)
        dataset = load_dataset(hf_name, subset, split=split, cache_dir=cache_dir)
        datasets_iter.append((subset, dataset))
    return datasets_iter


