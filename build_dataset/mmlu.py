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

    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--repo_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    return args


def main(args):
    """
    Loads the MMLU test dataset and associates each question with a randomly selected
    audio file as an interference signal. Then, pushes the modified dataset to the Hugging Face Hub.

    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - audio_path (str): Directory containing .wav audio files.
            - repo_name (str): Hugging Face repository to create and push the dataset.
            - seed (int): Random seed to ensure reproducibility.
    """
    random.seed(args.seed)

    mmlu = load_dataset("cais/mmlu", "all")["test"]
    question = mmlu["question"]
    subject = mmlu["subject"]
    choices = mmlu["choices"]
    answer = mmlu["answer"]

    assert len(question) == len(subject)
    assert len(question) == len(choices)
    assert len(question) == len(answer)

    dataset_size = len(answer)
    
    audio_files = [str(audio_file) for audio_file in Path(args.audio_path).rglob('*.wav')]

    sub_audio_files = random.sample(audio_files, dataset_size)

    interference_mmlu = Dataset.from_dict(
        {
            "audio": sub_audio_files,
            "question": question,
            "subject": subject,
            "choices": choices,
            "answer": answer
        }
    ).cast_column("audio", Audio())

    dataset_dict = {
        "test": interference_mmlu
    }
    interference_mmlu_dict = DatasetDict(dataset_dict)

    create_repo(args.repo_name, repo_type="dataset", private=True)
    interference_mmlu_dict.push_to_hub(args.repo_name)

    
if __name__ == "__main__":
    args = parse_args()
    main(args)

