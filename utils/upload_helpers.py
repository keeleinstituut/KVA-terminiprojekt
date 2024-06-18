import json
import re


def divide_chunks(l: list, n: int):
    """
    Divides a list into successive n-sized chunks.

    Args:
        - l (list): The list to be divided into chunks.
        - n (int): The size of each chunk.

    Yields:
        list: A chunk of the list of size n.

    Example:
        >>> list(divide_chunks([1, 2, 3, 4, 5, 6], 2))
        [[1, 2], [3, 4], [5, 6]]
    """
    for i in range(0, len(l), n):
        yield l[i: i + n]


def join_jsons(term_json_list: list) -> str:
    merged_list = []
    for term_json in term_json_list:
        merged_list.extend(json.loads(term_json))

    return json.dumps(merged_list)


def reformat_text(text: str) -> str:
    """
    Joins lines that don't end with a period or other punctuation marks denoting the end of sentence (incl. numbers).
    Also, replace consecutive empty lines with a single newline character.

    Parameters:
    - text (str): The text to be reformatted.

    Returns:
    - str: The reformatted text.
    """

    rows = text.split('\n')
    full_text = ''
    last_char = ''

    for row in rows:
        stripped = row.strip()
        start_delimiter = ''
        end_delimiter = ''

        if full_text:
            last_char = full_text[-1]

        if stripped == '' and last_char in ['', '\n']:
            continue

        # Single line break
        if stripped == '' and last_char != '\n':
            end_delimiter = '\n'

        # Uppercased or title-cased strings or strings not containing any alphabetic characters (Latin)
        elif stripped.isupper() or stripped.istitle() or stripped.upper().isupper() == False:
            end_delimiter = '\n'
            if last_char not in ['', '\n']:
                start_delimiter = '\n'

        elif last_char not in '0123456789.?!)' and last_char != '\n':
            start_delimiter = ' '

        elif last_char and last_char != '\n':
            start_delimiter = '\n'

        full_text += f'{start_delimiter}{stripped}{end_delimiter}'

    return full_text


def normalized_input_lists(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub('(,|;)', '/+/', text)
    return [kw.strip().lower() for kw in text.split('/+/')]
