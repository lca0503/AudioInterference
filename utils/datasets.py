from typing import Any, Dict

from datasets import load_dataset


def get_dataset(task_id: str, task_split: str, mitigate_prompt: bool):
    """
    Load and preprocess a dataset for a given task.

    Args:
        task_id (str): Identifier of the dataset.
        task_split (str): Dataset split to load.
        mitigate_prompt (bool): Whether to prepend an instruction to mitigate irrelevant prompts.

    Returns:
        datasets.Dataset: Preprocessed dataset ready for inference.
    """
    try:
        dataset = load_dataset(task_id)[task_split]
    except:
        raise NotImplementedError(f"Unsupported task: {task_id}")

    preprocessed_dataset = dataset.map(
        lambda sample: preprocess_query(sample, task_id, mitigate_prompt),
        load_from_cache_file=False
    )

    return preprocessed_dataset


def format_subject(subject: str) -> str:
    """
    Format subject string by replacing underscores with spaces.

    Args:
        subject (str): Subject string with underscores.

    Returns:
        str: Formatted subject string with spaces.
    """
    return " ".join(subject.split("_"))


def preprocess_query(sample: Dict[str, Any], task_id: str, mitigate_prompt: bool) -> Dict[str, Any]:
    """
    Preprocess a single dataset sample by constructing a query prompt.

    Args:
        sample (dict): A single dataset entry containing question, choices, etc.
        task_id (str): Dataset identifier to determine preprocessing logic.
        mitigate_prompt (bool): Whether to prepend a mitigation instruction.

    Returns:
        dict: Updated sample with an added "query" field.
    """
    if "mmlu" in task_id:
        subject = sample["subject"]
        question = sample["question"]
        choices = sample["choices"]
        sample["query"] = f"Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question.strip()}\n"
        choice_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices):
            sample["query"] += f"\n{choice_labels[i]}) {choice}"
        
    elif "gsm8k" in task_id:
        question = sample["question"]
        sample["query"] = f"Answer the following question. The last line of your response should be of the following format: 'Answer: ...'. Think step by step before answering.\n\nQuestion: {question}\n"
            
    elif "arc" in task_id:
        question = sample["question"]
        choices = sample["choices"]
        
        sample["query"] = f"Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step before answering.\n\n{question.strip()}\n"
        choice_labels = ["A", "B", "C", "D", "E"]
        valid_choices = choices['text']
        for i, choice in enumerate(valid_choices):
            sample["query"] += f"\n{choice_labels[i]}) {choice}"

    else:
        raise NotImplementedError(f"Unsupported task: {task_id}")

    if mitigate_prompt:
        sample["query"] = "Focus on the text or audio that contains useful information.\n" + sample["query"]
        
    return sample
