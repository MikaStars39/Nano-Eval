import argparse
from dataclasses import dataclass

@dataclass
class NanoEvalArgs:
    dataset: str
    cache_dir: str
    k: int
    f_out: str
    model: str
    prompt_format: str
    system_prompt: str
    max_concurrency: int
    max_new_tokens: int
    eval_temperature: float
    eval_top_p: float

def parse_args() -> NanoEvalArgs:
    parser = argparse.ArgumentParser(description="NanoEval Arguments")
    parser.add_argument("--dataset", type=str, default="aime2024")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--f_out", type=str, default="output.jsonl")
    parser.add_argument("--model", type=str, default="Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--prompt_format", type=str, default="slime")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max_concurrency", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--eval_top_p", type=float, default=1.0)