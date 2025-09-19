import json
import os


def save_json(filename, data):
    """
    Saves a Python object as a formatted JSON file.

    Args:
        filename (str): The path where the JSON file will be saved.
        data (Any): The Python object (e.g., dict or list) to save.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

        
def save_jsonl(filename, data):
    """
    Saves a list of Python dictionaries to a JSON Lines (.jsonl) file.

    Args:
        filename (str): The path where the JSONL file will be saved.
        data (list[dict]): A list of dictionaries to write, one per line.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            
def extract_json(filename):
    """
    Loads data from a JSON file.

    Args:
        filename (str): The path to the JSON file to read.

    Returns:
        Any: The parsed Python object from the JSON file (typically a dict or list).
    """
    with open(filename, "r", encoding='utf-8') as f:
        return json.load(f)

    
def extract_jsonl(filename):
    """
    Loads data from a JSON Lines (.jsonl) file.

    Args:
        filename (str): The path to the JSONL file to read.

    Returns:
        list[dict]: A list of dictionaries parsed from each line in the file.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

    
def get_all_jsonl_files(directory):
    """
    Recursively retrieves all .jsonl files in a directory.

    Args:
        directory (str): The root directory to search.

    Returns:
        list[str]: A list of file paths to all .jsonl files found within the directory.
    """
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith('.jsonl')
    ]
