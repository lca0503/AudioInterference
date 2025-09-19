import random
from argparse import ArgumentParser
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo


def parse_args():
    """
    Parses command-line arguments for audio path, repository name, and seed.

    Returns:
        argparse.Namespace: An object containing:
            - audio_path (str): Path to directory containing audio files.
            - repo_name (str): Hugging Face Hub dataset repository name.
            - seed (int): Random seed for reproducibility.
    """
    parser = ArgumentParser()

    parser.add_argument("--audio_path", type=str, default="./ESC-50/audio/")
    parser.add_argument("--repo_name", type=str, default="lca0503/gsm8k")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    return args


def main(args):
    """
    Loads the GSM8K test dataset and associates each question with a randomly selected
    audio file as an interference signal. Then, pushes the modified dataset to the Hugging Face Hub.

    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - audio_path (str): Directory containing .wav audio files.
            - repo_name (str): Hugging Face repository to create and push the dataset.
            - seed (int): Random seed to ensure reproducibility.
    """
    random.seed(args.seed)
    
    gsm8k = load_dataset("openai/gsm8k", "main")["test"]
    question = gsm8k["question"]
    answer = gsm8k["answer"]

    assert len(question) == len(answer)

    dataset_size = len(answer)

    audio_files = [str(audio_file) for audio_file in Path(args.audio_path).rglob('*.wav')]
    sub_audio_files = random.sample(audio_files, dataset_size)

    interference_gsm8k = Dataset.from_dict(
        {
            "audio": sub_audio_files,
            "question": question,
            "answer": answer
        }
    ).cast_column("audio", Audio())
    
    dataset_dict = {
        "test": interference_gsm8k
    }
    interference_gsm8k_dict = DatasetDict(dataset_dict)

    create_repo(args.repo_name, repo_type="dataset", private=True)
    interference_gsm8k_dict.push_to_hub(args.repo_name)


    
if __name__ == "__main__":
    args = parse_args()
    main(args)

