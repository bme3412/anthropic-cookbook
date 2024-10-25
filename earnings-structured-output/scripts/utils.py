# File: scripts/utils.py

import os
from typing import List

def list_transcript_files(directory: str) -> List[str]:
    """
    Lists all transcript files in the specified directory.
    Assumes transcripts are in .txt format.
    """
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.txt')]

def load_transcript(path: str) -> str:
    """
    Loads the content of a transcript file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()
