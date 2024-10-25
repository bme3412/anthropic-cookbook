# File: scripts/models.py

from pydantic import BaseModel
from typing import List, Optional, Dict

class ExecutiveComment(BaseModel):
    speaker: str
    timestamp: str
    content: str

class SalesInfo(BaseModel):
    amount: int
    currency: str
    growth_over_market: str
    notes: str

class RegionalPerformance(BaseModel):
    impact: str
    effect_on_growth: str

class FinancialHighlights(BaseModel):
    quarter: str
    sales: SalesInfo
    profitability: str
    regional_performance: Dict[str, RegionalPerformance]

class Participant(BaseModel):
    name: str
    role: str

class Participants(BaseModel):
    operator: Participant
    investor_relations: Participant
    executives: List[Participant]

class ForwardLookingStatements(BaseModel):
    text: str
    reference_page: str

class Disclaimers(BaseModel):
    recording_notice: str
    forward_looking_statements: ForwardLookingStatements

class PresentationMaterials(BaseModel):
    availability_date: str
    url: str
    description: str

class CallDetails(BaseModel):
    duration_minutes: int
    qa_instructions: str

class QnASessionDetails(BaseModel):
    after_executives: bool
    duration_minutes: int
    rules: str

class QnASession(BaseModel):
    scheduled: bool
    details: QnASessionDetails

class TranscriptContent(BaseModel):
    call_date: str
    start_time: str
    participants: Participants
    disclaimers: Disclaimers
    presentation_materials: PresentationMaterials
    call_details: CallDetails
    executive_comments: List[ExecutiveComment]
    financial_highlights: FinancialHighlights
    qna_session: QnASession

class TranscriptSchema(BaseModel):
    ticker: str
    download_date: str
    transcript: TranscriptContent
