import copy
import re
from collections import Counter


def find_numbers(x):
    """
    Extract all numbers from a given string using regex.

    Args:
        x (str): Input string.

    Returns:
        list of str: List of number substrings found in the text.
    """
    numbers = re.compile(
        r'-?[\d,]*\.?\d+',
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    
    return numbers


def find_number(x, answer_delimiter="answer"):
    """
    Find the first relevant number in a string, optionally after a delimiter.

    Args:
        x (str): Input string containing numbers.
        answer_delimiter (str): Optional keyword used to isolate the answer region (default: "answer").

    Returns:
        str: The extracted number as a string, or an empty string if none found.
    """
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ""


def remove_comma(x):
    """
    Remove commas from a string (useful for normalizing numeric strings).

    Args:
        x (str): Input string.

    Returns:
        str: String with commas removed.
    """
    x = x.replace(', ', ',')
    x = x.replace(',', '')
    return x
