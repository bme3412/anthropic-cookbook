# data_ingestion/models.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Speaker(BaseModel):
    name: str
    role: Optional[str] = None

class TranscriptSection(BaseModel):
    speaker: Speaker
    text: str
    timestamp: Optional[datetime] = None

class EarningsCallTranscript(BaseModel):
    company: str
    ticker: str
    earnings_date: datetime
    transcript: List[TranscriptSection]
    raw_text: Optional[str] = None  # Store raw transcript if needed
