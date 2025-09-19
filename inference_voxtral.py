import io
import os
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import (AudioChunk, RawAudio,
                                                       TextChunk, UserMessage)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tqdm import tqdm
from transformers import AutoProcessor, set_seed
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest

from utils.common import save_jsonl
from utils.datasets import get_dataset


def parse_args() -> Namespace:
    """
    Parse command-line arguments for running Voxtral inference.

    Returns:
        argparse.Namespace: Parsed arguments including task_id, task_split,
        task_type, model_id, output_path, seed, and whether to use
        prompt mitigation.
    """
    parser = ArgumentParser(description="Inference with Voxtral")

    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--task_split", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


def get_inputs(
    sample: Dict[str, Any],
    task_type: str,
    sampling_rate: int,
    tokenizer: MistralTokenizer,
    model_id: str
) -> Tuple[List[int], Optional[List[Tuple[np.ndarray, int]]]]:
    """
    Prepare model input prompt and audio data for a given sample.

    Args:
        sample (dict): A single dataset entry containing query and optionally audio.
        task_type (str): Task type, either 'text_bench' or 'text_bench_interference'.
        sampling_rate (int): Target audio sampling rate.
        tokenizer (MistralTokenizer): Tokenizer for encoding prompts.
        model_id (str): Model identifier.

    Returns:
        tuple:
            - List[int]: Encoded token IDs for the prompt.
            - Optional[List[Tuple[np.ndarray, int]]]: List of audio arrays with sampling rate,
              or None if no audio is required.
    """
    query = sample["query"]
    if task_type == "text_bench":
        query_chunk = TextChunk(text=query)
        messages = [UserMessage(content=[query_chunk])]
        audios = None
    else:
        audios = [
            librosa.resample(
                sample["audio"]["array"],
                orig_sr=sample["audio"]["sampling_rate"],
                target_sr=sampling_rate
            )
        ]
        
        wav_bytes = []
        for arr in audios:
            bio = io.BytesIO()
            sf.write(bio, arr, sampling_rate, format="WAV")
            wav_bytes.append(bio.getvalue())

        audios = [
            Audio.from_bytes(byte) for byte in wav_bytes 
        ]
        audio_chunks = [
            AudioChunk(input_audio=RawAudio.from_audio(audio)) for audio in audios
        ]
        query_chunk = TextChunk(text=query)
        messages = [UserMessage(content=[*audio_chunks, query_chunk])]
        
    req = ChatCompletionRequest(messages=messages, model=model_id)
    tokens = tokenizer.encode_chat_completion(req)
    prompt, audios = tokens.tokens, tokens.audios
    
    audios = [(au.audio_array, au.sampling_rate) for au in audios]

    return prompt, audios

    
def generate_responses(
    model_id: str,
    task_id: str,
    task_split: str,
    task_type: str,
    mitigate_prompt: bool,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Run inference with the Voxtral model on a dataset and collect responses.

    Args:
        model_id (str): Hugging Face model ID.
        task_id (str): Dataset identifier.
        task_split (str): Dataset split.
        task_type (str): Task type, either 'text_bench' or 'text_bench_interference'.
        mitigate_prompt (bool): Whether to apply prompt mitigation.
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

    engine_args = EngineArgs(
        model=model_id,
        max_model_len=8192,
        limit_mm_per_prompt={"audio": audio_count[task_type]},
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        gpu_memory_utilization=0.9,
        seed=seed,
    )
    tokenizer = MistralTokenizer.from_hf_hub(model_id)


    llm = LLM(**asdict(engine_args))
        
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=1024,
        seed=seed,
    )
    
    PRINT_FLAG = True
    batch_inputs = []
    for idx, sample in enumerate(tqdm(dataset, desc="Prepare Inputs")):
        prompt, audios = get_inputs(
            sample=sample,
            task_type=task_type,
            sampling_rate=16000,
            tokenizer=tokenizer,
            model_id=model_id,
        )
        if PRINT_FLAG:
            print(prompt)
            PRINT_FLAG = False

        if audio_count[task_type] == 0:
            batch_inputs.append({
                "prompt_token_ids": prompt
            })
        else:
            batch_inputs.append({
                "prompt_token_ids": prompt,
                "multi_modal_data": {"audio": audios}
            })
    
    lora_request = None

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
    Main execution function: runs inference with Voxtral and saves results.

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
        seed=args.seed
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_jsonl(args.output_path, results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
