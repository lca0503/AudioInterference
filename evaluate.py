import re
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

from utils.common import extract_jsonl
from utils.const import subject2category, subject2subcategory
from utils.eval_common import (MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
                               MULTILINGUAL_ANSWER_REGEXES,
                               normalize_extracted_answer, normalize_response)
from utils.eval_specify import find_number, remove_comma


def parse_args() -> Namespace:
    """
    Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments including input_path and task_id.
    """
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--scs", action="store_true")
        
    return parser.parse_args()


def score_mc_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score multiple-choice (MC) results by extracting the predicted answer
    and comparing it with the ground truth.

    Args:
        results (list[dict]): List of evaluation results, each containing
            "response" and "answer".

    Returns:
        list[dict]: Updated results with a new "score" field (1.0 or 0.0).
    """
    choices = ["A", "B", "C", "D"]
    for result in results:
        response = normalize_response(result["response"])
        extracted_answer = None
        for pattern in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(pattern)
            match = re.search(regex, response)
            if match:
                extracted_answer = normalize_extracted_answer(match.group(1))
                extracted_answer = extracted_answer.upper()
                break
        if extracted_answer is None: 
            print("Warning: Fail to extract the answer.", response)
        if type(result["answer"]) == int:
            result["answer"] = choices[result["answer"]]
        result["score"] = 1.0 if extracted_answer == result["answer"] else 0.0
    return results


def score_num_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score numeric results (e.g., GSM8K) by extracting numbers
    from the response and comparing them to the ground truth.

    Args:
        results (list[dict]): List of evaluation results.

    Returns:
        list[dict]: Updated results with a "score" field.
    """
    for result in results:
        response = result["response"].lower()
        answer = result["answer"]
        answer = answer.split('### ')[-1].rstrip()
        pred = find_number(remove_comma(response))
        answer = remove_comma(answer)
        try:
            result["score"] = 1.0 if float(pred) == float(answer) else 0.0
        except ValueError:
            result["score"] = 0.0
    return results


def score_mc_results_scs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score multiple-choice (MC) results  with self-consistency by extracting 
    the predicted answer and comparing it with the ground truth.

    Args:
        results (list[dict]): List of evaluation results, each containing
            "response" and "answer".

    Returns:
        list[dict]: Updated results with a new "score" field (1.0 or 0.0).
    """
    choices = ["A", "B", "C", "D"]
    for result in results:
        extracted_answers = []
        for response in result["response"]:
            response = normalize_response(response)
            extracted_answer = None
            for pattern in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(pattern)
                match = re.search(regex, response)
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    extracted_answer = extracted_answer.upper()
                    break
            if extracted_answer is None: 
                print("Warning: Fail to extract the answer.", response)
            if type(result["answer"]) == int:
                result["answer"] = choices[result["answer"]]
            extracted_answers.append(extracted_answer)
        final_answer = Counter(extracted_answers).most_common(1)[0][0]
        result["score"] = 1.0 if final_answer == result["answer"] else 0.0
    return results


def score_num_results_scs(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score numeric results with self-consistency (e.g., GSM8K) by extracting 
    numbers from the response and comparing them to the ground truth.

    Args:
        results (list[dict]): List of evaluation results.

    Returns:
        list[dict]: Updated results with a "score" field.
    """
    for result in results:
        answer = result["answer"]
        answer = answer.split('### ')[-1].rstrip()
        answer = remove_comma(answer)
        extracted_answers = []
        for response in result["response"]:
            response = response.lower()
            extracted_answer = find_number(remove_comma(response))
            extracted_answers.append(extracted_answer)

        final_answer = Counter(extracted_answers).most_common(1)[0][0]
        try:
            result["score"] = 1.0 if float(final_answer) == float(answer) else 0.0
        except ValueError:
            result["score"] = 0.0
            
    return results


def compute_mmlu_accuracy(results: List[Dict[str, Any]]) -> None:
    """
    Compute accuracy for MMLU tasks, grouped by subject category.

    Args:
        results (list[dict]): Evaluation results with "subject" and "score".

    Returns:
        None
    """
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    group_fn = lambda r: subject2category.get(r["subject"], "other")
    
    for result in results:
        group = group_fn(result)
        group_correct[group] += result["score"]
        group_total[group] += 1

    for group in group_correct:
        correct = group_correct[group]
        total = group_total[group]
        accuracy = correct / total if total else 0
        print(f"{group} Accuracy: {accuracy:.2%} ({int(correct)}/{total})")

        
def compute_accuracy(results: List[Dict[str, Any]]) -> None:
    """
    Compute overall accuracy (no grouping).

    Args:
        results (list[dict]): Evaluation results with "score".

    Returns:
        None
    """
    correct = 0
    total = 0
    
    for result in results:
        correct += result["score"]
        total += 1

    accuracy = correct / total if total else 0
    print(f"Accuracy: {accuracy:.2%} ({int(correct)}/{total})")
    

def main(args: Namespace) -> None:
    """
    Main entry point for evaluation.
    Loads results, applies task-specific scoring, and prints accuracy metrics.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    """
    results = extract_jsonl(args.input_path)    

    if "mmlu" in args.task_id:
        if args.scs:
            results = score_mc_results_scs(results)
        else:
            results = score_mc_results(results)
        compute_accuracy(results)
    elif "arc" in args.task_id:
        if args.scs:
            results = score_mc_results_scs(results)
        else:
            results = score_mc_results(results)
        compute_accuracy(results)
    elif "gsm8k" in args.task_id:
        if args.scs:
            results = score_num_results_scs(results)
        else:
            results = score_num_results(results)
        compute_accuracy(results)
    else:
        raise NotImplementedError(f"Unsupported task: {args.task_id}")

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
