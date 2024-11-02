# models.py
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class FinancialMetric(BaseModel):
    metric_name: str
    value: float
    period: str
    year: int
    quarter: Optional[int] = None  # Made optional with default None
    description: Optional[str] = None

class KPI(BaseModel):
    name: str
    value: str
    context: Optional[str]

class OperationalMetric(BaseModel):
    metric_name: str
    value: str  # Changed from float to str to handle percentage ranges
    description: Optional[str]

class RiskFactor(BaseModel):
    risk_name: str
    description: str

class StrategicInitiative(BaseModel):
    initiative_name: str
    progress: str
    impact: Optional[str]

class EarningsAnalysis(BaseModel):
    company: str
    date: datetime
    metrics: List[FinancialMetric] = []
    kpis: List[KPI] = []
    operational_metrics: List[OperationalMetric] = []
    risk_factors: List[RiskFactor] = []
    strategic_initiatives: List[StrategicInitiative] = []
    summary: str
    key_highlights: List[str]

class TokenUsageTracker:
    """Track token usage and costs across different models"""
    
    PRICE_PER_1K_TOKENS = {
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'text-embedding-3-small': {'input': 0.00002, 'output': 0.00002}
    }

    def __init__(self):
        self.usage = {
            'gpt-4o-mini': {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0},
            'text-embedding-3-small': {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
        }
        
        try:
            import tiktoken
            self.encoders = {
                'gpt-4o-mini': tiktoken.get_encoding("cl100k_base"),
                'text-embedding-3-small': tiktoken.get_encoding("cl100k_base")
            }
        except Exception as e:
            print(f"Error initializing tokenizers: {e}")
            self.encoders = {model: None for model in self.usage.keys()}

    def count_tokens(self, text: str, model: str) -> int:
        try:
            if self.encoders[model]:
                return len(self.encoders[model].encode(text))
            return len(text.split())  # Fallback to word count
        except Exception as e:
            print(f"Error counting tokens for {model}: {e}")
            return 0

    def add_usage(self, model: str, input_tokens: int, output_tokens: int = 0):
        if model not in self.usage:
            return
            
        self.usage[model]['input_tokens'] += input_tokens
        self.usage[model]['output_tokens'] += output_tokens
        
        input_cost = (input_tokens / 1000) * self.PRICE_PER_1K_TOKENS[model]['input']
        output_cost = (output_tokens / 1000) * self.PRICE_PER_1K_TOKENS[model]['output']
        self.usage[model]['cost'] += input_cost + output_cost

    def get_total_usage(self) -> dict:
        total_cost = sum(model_usage['cost'] for model_usage in self.usage.values())
        total_tokens = sum(
            model_usage['input_tokens'] + model_usage['output_tokens']
            for model_usage in self.usage.values()
        )
        return {
            'total_cost': round(total_cost, 4),
            'total_tokens': total_tokens,
            'detailed_usage': self.usage
        }