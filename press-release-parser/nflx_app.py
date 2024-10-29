from typing import List, Optional, Dict
from datetime import datetime
import json
from anthropic import Anthropic
import backoff
from pydantic import BaseModel, Field, field_validator
import os

class RegionalPerformance(BaseModel):
    name: str = Field(..., description="Region name (e.g., 'LatAm')")
    performance: str = Field(..., description="Overall performance description")
    revenue_growth: Optional[float] = None
    quote: str

class ContentItem(BaseModel):
    type: str = Field(..., description="Type of content (Series/Film/Live/Game)")
    titles: List[str] = Field(default_factory=list)

class GuidanceMetrics(BaseModel):
    revenue: Optional[Dict[str, float]] = Field(None, description="Revenue guidance range in billions")
    revenue_growth: str = Field(..., description="Expected revenue growth range")
    membership_growth: Optional[str] = None
    margin_guidance: Optional[str] = None
    quote: str

class AdvertisingMetrics(BaseModel):
    signup_share: str = Field(..., description="Ad tier share of signups")
    growth_rate: str = Field(..., description="Quarter over quarter growth")
    engagement: str = Field(..., description="Engagement metrics vs non-ad tier")
    revenue_projection: str = Field(..., description="Revenue growth projection")
    quote: str

class NetflixEarningsAnalysis(BaseModel):
    # Document metadata
    symbol: str
    quarter: int
    year: int
    date: datetime
    
    # Key metrics
    viewing_hours: float = Field(..., description="Daily viewing hours per member")
    engagement_growth: str
    revenue: Optional[float] = None
    revenue_growth: float
    operating_margin_improvement: Optional[float] = None
    
    # Business segments
    regional_performance: List[RegionalPerformance]
    
    # Content pipeline
    upcoming_content: List[ContentItem]
    
    # Forward looking
    guidance: GuidanceMetrics
    
    # Advertising business
    advertising: AdvertisingMetrics
    
    # Strategic priorities
    strategic_priorities: List[str]
    
    # Notable quotes
    key_quotes: Dict[str, str]

    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        return v

    def dict_with_formatted_date(self):
        d = self.model_dump()
        d['date'] = self.date.strftime("%Y-%m-%d %H:%M:%S")
        return d

class NetflixTranscriptAnalyzer:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _clean_json_response(self, response_text: str) -> str:
        """Clean the response text to get valid JSON"""
        clean_text = response_text.strip()
        if clean_text.startswith('```json'):
            clean_text = clean_text[7:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
        return clean_text.strip()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def analyze_transcript(self, content: str, metadata: Dict) -> NetflixEarningsAnalysis:
        try:
            messages = [
                {
                    "role": "user",
                    "content": f"""Analyze this Netflix Q3 2024 earnings call transcript and extract information in JSON format. Focus on these key areas:

1. Viewing & Engagement:
- Daily viewing hours per member (look for specific numbers)
- Engagement growth trends
- Total viewing hours mentioned

2. Financial Metrics:
- Revenue guidance
- Revenue growth expectations
- Operating margin improvements

3. Regional Performance:
- Focus on specific regions discussed (e.g., LatAm performance)
- Include revenue growth and key metrics by region

4. Content Pipeline:
For each content type (Series/Film/Live/Game), extract:
- Upcoming titles mentioned
- Expected release timeframes
- Strategic importance

5. Advertising Business:
- Ad tier adoption metrics
- Growth rates
- Engagement comparisons
- Revenue projections

6. Strategic Priorities:
- Key initiatives mentioned
- Investment priorities
- Growth strategies

Include relevant quotes for each key point.

Transcript to analyze:

{content}

Provide a JSON response matching this structure:
{{
    "viewing_hours": float,
    "engagement_growth": string,
    "revenue_growth": float,
    "operating_margin_improvement": float,
    "regional_performance": [
        {{
            "name": string,
            "performance": string,
            "revenue_growth": float,
            "quote": string
        }}
    ],
    "upcoming_content": [
        {{
            "type": string,
            "titles": [string]
        }}
    ],
    "guidance": {{
        "revenue_growth": string,
        "membership_growth": string,
        "margin_guidance": string,
        "quote": string
    }},
    "advertising": {{
        "signup_share": string,
        "growth_rate": string,
        "engagement": string,
        "revenue_projection": string,
        "quote": string
    }},
    "strategic_priorities": [string],
    "key_quotes": {{
        "topic": "quote"
    }}
}}"""
                }
            ]

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                messages=messages
            )
            
            try:
                cleaned_response = self._clean_json_response(response.content[0].text)
                analysis_dict = json.loads(cleaned_response)
                
                # Add metadata
                analysis_dict.update(metadata)
                
                return NetflixEarningsAnalysis(**analysis_dict)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing Claude response: {str(e)}")
                print("Raw response:", response.content[0].text)
                print("Cleaned response:", cleaned_response)
                raise
            
        except Exception as e:
            print(f"Error analyzing transcript: {str(e)}")
            raise

def format_results(analysis: dict) -> None:
    """Print formatted analysis results"""
    print("\nNetflix Q3 2024 Earnings Analysis")
    print("="*40)
    
    print("\nKey Metrics:")
    print(f"• Daily Viewing: {analysis.get('viewing_hours', 'N/A')} hours per member")
    print(f"• Revenue Growth: {analysis.get('revenue_growth', 'N/A')}%")
    print(f"• Operating Margin Improvement: {analysis.get('operating_margin_improvement', 'N/A')} points")
    
    if analysis.get('upcoming_content'):
        print("\nUpcoming Content Pipeline:")
        for content in analysis['upcoming_content']:
            print(f"\n{content['type']}:")
            for title in content['titles']:
                print(f"• {title}")
    
    if analysis.get('advertising'):
        print("\nAdvertising Business:")
        ad = analysis['advertising']
        print(f"• Sign-up Share: {ad.get('signup_share', 'N/A')}")
        print(f"• Growth Rate: {ad.get('growth_rate', 'N/A')}")
        print(f"• Engagement: {ad.get('engagement', 'N/A')}")
        print(f"• Revenue Projection: {ad.get('revenue_projection', 'N/A')}")
    
    if analysis.get('strategic_priorities'):
        print("\nStrategic Priorities:")
        for priority in analysis['strategic_priorities']:
            print(f"• {priority}")
    
    if analysis.get('key_quotes'):
        print("\nNotable Quotes:")
        for topic, quote in analysis['key_quotes'].items():
            print(f"\n{topic}:")
            print(f'"{quote}"')  # Fixed quote formatting

def process_transcript(file_path: str, api_key: str) -> dict:
    """Process a single Netflix earnings transcript"""
    analyzer = NetflixTranscriptAnalyzer(api_key)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    metadata = {
        'symbol': data['symbol'],
        'quarter': data['quarter'],
        'year': data['year'],
        'date': data['date']
    }
    
    analysis = analyzer.analyze_transcript(data['content'], metadata)
    return analysis.dict_with_formatted_date()

def save_analysis(analysis: dict, output_path: str):
    """Save analysis results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    FILE_PATH = "NFLX.json"
    OUTPUT_PATH = "netflix_earnings_analysis.json"
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    if not API_KEY:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    try:
        print("Analyzing Netflix Q3 2024 earnings transcript...")
        analysis = process_transcript(FILE_PATH, API_KEY)
        
        print("Saving detailed analysis...")
        save_analysis(analysis, OUTPUT_PATH)
        
        format_results(analysis)
        print(f"\nFull analysis saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise