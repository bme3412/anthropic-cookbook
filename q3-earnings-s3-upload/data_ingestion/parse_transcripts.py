# data_ingestion/parse_transcripts.py

import re
from datetime import datetime
from typing import List
from .models import EarningsCallTranscript, TranscriptSection, Speaker
from utils.logger import logger

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError as ve:
        logger.error(f"Date parsing error: {ve}")
        # Handle the error as needed, e.g., set to None or a default date
        return None
    
def parse_raw_transcript(raw_text: str, company: str, ticker: str, earnings_date: str) -> EarningsCallTranscript:
    """
    Parses raw transcript text into a structured EarningsCallTranscript model.
    Assumes transcripts have sections starting with speaker names.
    """
    sections = []
    lines = raw_text.splitlines()
    current_speaker = None
    current_text = []

    # Adjust the regex pattern based on the actual transcript format
    speaker_pattern = re.compile(r"^(?P<name>[A-Za-z\s]+):\s*(?P<text>.*)")

    for line in lines:
        match = speaker_pattern.match(line)
        if match:
            if current_speaker and current_text:
                sections.append(
                    TranscriptSection(
                        speaker=current_speaker,
                        text=" ".join(current_text),
                        timestamp=None  # If timestamps are available, parse them
                    )
                )
                current_text = []
            speaker_name = match.group("name").strip()
            speaker_text = match.group("text").strip()
            current_speaker = Speaker(name=speaker_name)
            current_text.append(speaker_text)
        else:
            if current_speaker:
                current_text.append(line.strip())

    # Add the last section
    if current_speaker and current_text:
        sections.append(
            TranscriptSection(
                speaker=current_speaker,
                text=" ".join(current_text),
                timestamp=None
            )
        )

    # Parse the earnings_date correctly
    parsed_date = parse_date(earnings_date)
    if not parsed_date:
        # Handle the parsing failure as needed
        logger.warning(f"Earnings date parsing failed for transcript: {raw_text}")
        # You might choose to skip this transcript or set a default date

    # Create the EarningsCallTranscript model
    try:
        transcript_model = EarningsCallTranscript(
            company=company,
            ticker=ticker,
            earnings_date=parsed_date,
            transcript=sections,
            raw_text=raw_text
        )
    except Exception as e:
        logger.error(f"Error creating transcript model: {e}")
        transcript_model = EarningsCallTranscript(
            company=company,
            ticker=ticker,
            earnings_date=parsed_date if parsed_date else datetime.now(),
            transcript=sections,
            raw_text=raw_text
        )

    return transcript_model
