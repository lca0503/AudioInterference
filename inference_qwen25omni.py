import os
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, set_seed
from vllm import LLM, EngineArgs, SamplingParams

from utils.common import save_jsonl
from utils.datasets import get_dataset


def parse_args() -> Namespace:
    """
    Parse command-line arguments for running inference.

    Returns:
        argparse.Namespace: Parsed arguments including task_id, task_split,
        task_type, model_id, output_path, temperature, seed, and whether to
        use prompt mitigation.
    """
    parser = ArgumentParser(description="Inference with Qwen2.5-Omni series")

    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--task_split", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


def get_inputs(
    sample: Dict[str, Any],
    processor: AutoProcessor,
    task_type: str,
    sampling_rate: int
) -> Tuple[str, Optional[List[np.ndarray]]]:
    """
    Prepare model input prompt and audio data for a given sample.

    Args:
        sample (dict): A single dataset entry containing query and optionally audio.
        processor (AutoProcessor): Hugging Face processor for formatting input.
        task_type (str): Task type, either 'text_bench' or 'text_bench_interference'.
        sampling_rate (int): Target audio sampling rate.

    Returns:
        tuple: (prompt, audios) where prompt is the formatted input text
        and audios is a list of processed audio arrays or None.
    """
    query = sample["query"]
    if task_type == "text_bench":
        content = [
            {"type": "text", "text": query}
        ]
        audios = None
    else:
        content = [
            {"type": "audio", "audio_url": None},
            {"type": "text", "text": query},
        ]
        audios = [
            librosa.resample(
                sample["audio"]["array"],
                orig_sr=sample["audio"]["sampling_rate"],
                target_sr=sampling_rate,
            )
        ]
        
    prompt = processor.apply_chat_template(
        [{
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
         {
             "role": "user",
             "content": content
         }],
        add_generation_prompt=True,
        tokenize=False
    )
    return prompt, audios

    
def generate_responses(
    model_id: str,
    task_id: str,
    task_split: str,
    task_type: str,
    mitigate_prompt: bool,
    temperature: float,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Run inference with the Qwen model on a dataset and collect responses.

    Args:
        model_id (str): Hugging Face model ID.
        task_id (str): Dataset identifier.
        task_split (str): Dataset split.
        task_type (str): Task type, either 'text_bench' or 'text_bench_interference'.
        mitigate_prompt (bool): Whether to apply prompt mitigation.
        temperature (float): Sampling temperature for generation.
        seed (int): Random seed.

    Returns:
        list[dict]: A list of results containing metadata, prompts, responses,
        and ground-truth answers.
    """
    dataset = get_dataset(task_id, task_split, mitigate_prompt)

    audio_count = {
        "text_bench": 0,
        "text_bench_interference": 1,
    }

    processor = AutoProcessor.from_pretrained(model_id)
    
    engine_args = EngineArgs(
        model=model_id,
        max_model_len=8192,
        limit_mm_per_prompt={"audio": audio_count[task_type]},
        gpu_memory_utilization=0.9,
        seed=seed,
    )

    llm = LLM(**asdict(engine_args))
        
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=1024,
        seed=seed,
    )

    lora_request = None
    
    PRINT_FLAG = True
    batch_inputs = []
    for idx, sample in enumerate(tqdm(dataset, desc="Prepare Inputs")):
        prompt, audios = get_inputs(
            sample=sample,
            processor=processor,
            task_type=task_type,
            sampling_rate=16000,
        )
        if PRINT_FLAG:
            print(prompt)
            PRINT_FLAG = False

        if audio_count[task_type] == 0:
            batch_inputs.append({
                "prompt": prompt
            })
        else:
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": audios}
            })
            
    outputs = llm.generate(
        batch_inputs,
        sampling_params=sampling_params,
        lora_request=lora_request
    )

    results = []
    for sample, output in zip(dataset, outputs):
        generated_text = output.outputs[0].text.strip()
        results.append({
            "subject": sample.get("subject", ""),
            "task": sample.get("task", ""),
            "prompt": sample.get("prompt", ""),
            "query": sample["query"],
            "prefix": sample.get("prefix", ""),
            "choices": sample.get("choices", ""),
            "response": generated_text,
            "answer": sample["answer"],
        })
        
    return results


def main(args: Namespace) -> None:
    """
    Main execution function: runs inference and saves results.

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
        temperature=args.temperature,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_jsonl(args.output_path, results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
