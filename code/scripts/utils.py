from typing import TYPE_CHECKING, List, Optional, Tuple

import unidecode
from datasets import Dataset, NERDataset


def read_jsonl(
        filepath: str,
        encoding: Optional[str] = 'utf-8'
) -> 'Dataset':
    return NERDataset.from_jsonl(filepath, encoding)


def split_sentences(text: str) -> List[str]:
    return text.split('\n')


def get_offsets(
        text: str,
        tokens: List[str],
        start: Optional[int] = 0) -> List[int]:
    """Calculate char offsets of each tokens.

    Args:
        text (str): The string before tokenized.
        tokens (List[str]): The list of the string. Each string corresponds
            token.
        start (Optional[int]): The start position.
    Returns:
        (List[str]): The list of the offset.
    """
    text = unidecode.unidecode(text)
    offsets = []

    i = 0
    for token in tokens:
        if len(token)>1 and ord(token[0]) == 9601:
            token = token[1:]
        if len(token) == 1 and ord(token) == 9601:
            offsets.append(i+1)
            continue
        token = unidecode.unidecode(token)
        for j, char in enumerate(token):
            while char != text[i]:
                i += 1    
            if j == 0:
                offsets.append(i + start)
    return offsets

def get_sentence_offsets(text: str,
        tokens: List[str],
        start: Optional[int] = 0) -> List[int]:
    offsets = []
    i = 0
    text = unidecode.unidecode(text)
    for token in tokens:
        token = unidecode.unidecode(token)
        for j, char in enumerate(token):
            while char != text[i]:
                i += 1
            if j == 0:
                offsets.append(i + start)
    return offsets


def create_bio_tags(
        tokens: List[str],
        offsets: List[int],
        labels: List[Tuple[int, int, str]]) -> List[str]:
    """Create BI tags from Doccano's label data.

    Args:
        tokens (List[str]): The list of the token.
        offsets (List[str]): The list of the character offset.
        labels (List[Tuple[int, int, str]]): The list of labels. Each item in
            the list holds three values which are the start offset, the end
            offset, and the label name.
    Returns:
        (List[str]): The list of the BIO tag.
    """
    labels = sorted(labels)
    n = len(labels)
    i = 0
    prefix = 'B-'
    tags = []
    for token, token_start in zip(tokens, offsets):
        if len(token)>0 and ord(token[0]) == 9601:
            token = token[1:]
        token_end = token_start + len(token)
        if i >= n or token_end < labels[i][0]:
            tags.append('O')
        elif token_start > labels[i][1]:
            tags.append('O')
        elif str(labels[i][2]) == 'O':
            tags.append('O')
        else:
            tags.append(prefix + str(labels[i][2]))
            if labels[i][1] > token_end:
                prefix = 'I-'
            elif i < n:
                i += 1
                prefix = 'B-'
    return tags


class Token:
    def __init__(self, token: str, offset: int, i: int) -> None:
        self.token = token
        self.idx = offset
        self.i = i

    def __len__(self):
        return len(self.token)

    def __str__(self):
        return self.token
