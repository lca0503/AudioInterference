import base64
import io
import json
import os
import random
import time
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm

from utils.common import save_jsonl
from utils.datasets import get_dataset


def parse_args() -> Namespace:
    """
    Parse command-line arguments for running inference with Gemini.

    Returns:
        argparse.Namespace: Parsed arguments including task_id, task_split,
        task_type, model_id, output_path, seed, and whether to
        use prompt mitigation.
    """
    parser = ArgumentParser(description="Inference with Gemini")

    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--task_split", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


def load_existing_queries(output_path: str) -> Set[str]:
    """
    Load queries already processed from an existing output file.

    Args:
        output_path (str): Path to the JSONL output file.

    Returns:
        set[str]: Set of query strings already present in the output file.
    """
    if not os.path.exists(output_path):
        return set()
    with open(output_path, "r") as f:
        return {json.loads(line)["query"] for line in f if line.strip()}

    
def generate_responses(
    client: genai.Client,
    model_id: str,
    task_id: str,
    task_split: str,
    task_type: str,
    mitigate_prompt: bool,
    output_path: str
) -> None:
    """
    Run inference with Gemini on a dataset and append responses to output file.

    Args:
        client (genai.Client): Google GenAI client for interacting with Gemini.
        model_id (str): Model identifier for Gemini.
        task_id (str): Dataset identifier.
        task_split (str): Dataset split.
        task_type (str): Task type, either 'text_bench' or 'text_bench_interference'.
        mitigate_prompt (bool): Whether to apply prompt mitigation.
        output_path (str): Path to JSONL file where results will be appended.

    Returns:
        None
    """
    dataset = get_dataset(task_id, task_split, mitigate_prompt)
    seen_queries = load_existing_queries(output_path)

    queries = [
        sample.get("query", "")
        for sample in dataset
    ]
    start_index = next((i for i, q in enumerate(queries) if q not in seen_queries), len(dataset))

    with open(output_path, "a") as out_file:
        for idx in tqdm(range(start_index, len(dataset)), desc="Running inference"):
            sample = dataset[idx]
            query = queries[idx]

            contents = [query]
            if task_type != "text_bench":
                audio_arr = sample["audio"]["array"]
                sr_orig = sample["audio"]["sampling_rate"]
                audio_resamp = librosa.resample(audio_arr, orig_sr=sr_orig, target_sr=16000)
                bio = io.BytesIO()
                sf.write(bio, audio_resamp, 16000, format="WAV")
                audio_bytes = bio.getvalue()
                contents = [
                    types.Part.from_bytes(data=audio_bytes, mime_type='audio/wav'),
                    query
                ]
                
            max_retries = 10
            retry_delay = 5  # seconds
            response_text = ""

            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=0, responseModalities=["text"], seed=0)
                    )
                    response_text = response.text.strip()
                    if response_text == "":
                        raise ValueError("Error: Empty response received")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retrying ({attempt + 1}/{max_retries}) due to error: {e}")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed after {max_retries} attempts for sample {sample.get('id', '')}: {e}")
                        #print("Failed query:\n", contents)
                        if "NoneType" in str(e):
                            response_text = ""
                        else:
                            exit()
            result = {
                "subject": sample.get("subject", ""),
                "task": sample.get("task", ""),
                "prompt": sample.get("prompt", ""),
                "query": query,
                "choices": sample.get("choices", ""),
                "response": response_text,
                "answer": sample.get("answer", "")
            }

            out_file.write(json.dumps(result) + "\n")
            out_file.flush()

            
def main(args: Namespace) -> None:
    """
    Main execution function: runs inference with Gemini and saves results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    random.seed(args.seed)
    assert args.task_type in ["text_bench", "text_bench_interference"]

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    generate_responses(
        client=client,
        model_id=args.model_id,
        task_id=args.task_id,
        task_split=args.task_split,
        task_type=args.task_type,
        mitigate_prompt=args.mitigate_prompt,
        output_path=args.output_path,
    )

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
