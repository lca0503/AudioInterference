import os
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from desta import DeSTA25AudioModel
from tqdm import tqdm
from transformers import set_seed

from utils.common import save_jsonl
from utils.datasets import get_dataset


def parse_args() -> Namespace:
    """
    Parse command-line arguments for running inference with DeSTA25.

    Returns:
        argparse.Namespace: Parsed arguments including task_id, task_split,
        task_type, model_id, output_path, seed, and whether to
        use prompt mitigation.
    """
    parser = ArgumentParser(description="Inference with DeSTA25")

    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--task_split", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def generate_responses(
    model_id: str,
    task_id: str,
    task_split: str,
    task_type: str,
    mitigate_prompt: bool,
) -> List[Dict[str, Any]]:
    """
    Run inference with the DeSTA25 audio model on a dataset and collect responses.

    Args:
        model_id (str): Hugging Face model ID for DeSTA25.
        task_id (str): Dataset identifier.
        task_split (str): Dataset split.
        task_type (str): Task type, either 'text_bench' or 'text_bench_interference'.
        mitigate_prompt (bool): Whether to apply prompt mitigation.

    Returns:
        list[dict]: A list of results containing metadata, prompts, responses,
        and ground-truth answers.
    """
    model = DeSTA25AudioModel.from_pretrained(model_id)
    model.to("cuda")
    model.eval()

    dataset = get_dataset(task_id, task_split, mitigate_prompt)

    results = []
    for sample in tqdm(dataset, desc=f"Generating Response"):
        query = sample["query"]
        if args.task_type == "text_bench":
            messages = [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        else:
            array = sample["audio"]["array"]
            sr    = sample["audio"]["sampling_rate"]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
                sf.write(tmp_path, array, sr)
            messages = [
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{query}",
                    "audios": [{
                        "audio": tmp_path,
                        "text": None
                    }]
                }
            ]
        outputs = model.generate(
            messages=messages,
            do_sample=False,
            max_new_tokens=1024,
        )
        response = outputs.text[0]

        results.append({
            "subject": sample.get("subject", ""),
            "task": sample.get("task", ""),
            "prompt": sample.get("prompt", ""),
            "query": query,
            "choices": sample.get("choices", ""),
            "response": response,
            "answer": sample["answer"],
        })

    return results


def main(args: Namespace) -> None:
    """
    Main execution function: runs inference with DeSTA25 and saves results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    set_seed(args.seed)
    assert args.task_type in ["text_bench", "text_bench_interference"]

    results = generate_responses(
        model_id=args.model_id,
        task_id=args.task_id,
        task_split=args.task_split,
        task_type=args.task_type,
        mitigate_prompt=args.mitigate_prompt,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_jsonl(args.output_path, results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
