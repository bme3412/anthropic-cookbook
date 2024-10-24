from dotenv import load_dotenv
import os
from anthropic import Anthropic
import json
from typing import Dict
from datetime import datetime

class EarningsAnalyzer:
    def __init__(self):
        load_dotenv()
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-5-sonnet-latest"
        self.temperature = 0.7

    def generate_prompt(self, transcript: str, analysis_type: str) -> str:
        """Generate analysis prompts based on the analysis type."""
        prompts = {
            "earnings_analysis": """
    Analyze this earnings call transcript comprehensively, focusing on AI initiatives, financial metrics, and strategic insights.
    Return a structured JSON object with specific, quantitative details following this exact schema:

    {
        "financial_overview": {
            "key_metrics": {
                "revenue": {
                    "amount": "dollar amount",
                    "growth": "percentage with +/- sign",
                    "sequential_growth": "quarter-over-quarter growth",
                    "year_over_year": "year-over-year growth",
                    "source_quote": "exact quote from transcript"
                },
                "operating_income": {
                    "amount": "dollar amount",
                    "margin": "percentage",
                    "growth": "percentage with +/- sign",
                    "source_quote": "exact quote from transcript"
                },
                "eps": {
                    "value": "dollar amount per share",
                    "growth": "percentage with +/- sign",
                    "beat_miss": "difference from consensus",
                    "source_quote": "exact quote from transcript"
                },
                "cash_flow": {
                    "operating": "dollar amount",
                    "free_cash_flow": "dollar amount",
                    "cash_position": "total cash and equivalents",
                    "source_quote": "exact quote from transcript"
                }
            },
            "segment_performance": [
                {
                    "name": "segment name",
                    "revenue": "dollar amount",
                    "growth": "percentage with +/- sign",
                    "operating_margin": "percentage",
                    "key_products": ["list of key products"],
                    "highlights": ["key performance points"],
                    "source_quote": "exact quote from transcript"
                }
            ],
            "guidance": {
                "next_quarter": {
                    "revenue_range": "expected range in dollars",
                    "growth_range": "expected growth range",
                    "operating_margin": "expected margin range",
                    "key_drivers": ["specific growth drivers"],
                    "assumptions": ["key assumptions"],
                    "source_quote": "exact quote from transcript"
                },
                "full_year": {
                    "revenue_range": "expected range in dollars",
                    "growth_range": "expected growth range",
                    "operating_margin": "expected margin range",
                    "key_drivers": ["specific growth drivers"],
                    "assumptions": ["key assumptions"],
                    "source_quote": "exact quote from transcript"
                }
            }
        },
        "ai_initiatives": {
            "core_strategy": {
                "initiatives": [
                    {
                        "name": "initiative name",
                        "description": "detailed description",
                        "stage": "announced|in_development|launched",
                        "timeline": "specific timeline",
                        "investment_size": "dollar amount if mentioned",
                        "expected_impact": "quantified business impact",
                        "source_quote": "exact quote from transcript",
                        "key_metrics": ["relevant metrics or KPIs"]
                    }
                ],
                "key_focus_areas": ["specific strategic priorities"],
                "investment_summary": {
                    "total_investment": "dollar amount",
                    "timeframe": "investment period",
                    "priority_areas": ["investment priorities"]
                }
            },
            "products_and_features": {
                "current": [
                    {
                        "name": "product name",
                        "description": "detailed description",
                        "launch_date": "when launched",
                        "adoption_metrics": "user/customer numbers",
                        "market_position": "competitive positioning",
                        "revenue_impact": "revenue contribution",
                        "growth_rate": "growth metrics",
                        "source_quote": "exact quote from transcript"
                    }
                ],
                "planned": [
                    {
                        "name": "product name",
                        "description": "detailed description",
                        "target_launch": "expected launch timeline",
                        "development_stage": "current stage",
                        "target_market": "intended audience",
                        "expected_impact": "projected business impact",
                        "dependencies": ["key dependencies"],
                        "source_quote": "exact quote from transcript"
                    }
                ]
            },
            "partnerships": {
                "current": [
                    {
                        "partner": "partner name",
                        "description": "partnership details",
                        "start_date": "when initiated",
                        "scope": "partnership scope",
                        "deal_terms": "financial/business terms",
                        "strategic_value": "strategic importance",
                        "source_quote": "exact quote from transcript"
                    }
                ],
                "strategic_investments": [
                    {
                        "company": "company name",
                        "amount": "investment amount",
                        "stake": "ownership percentage",
                        "purpose": "strategic rationale",
                        "expected_returns": "expected benefits",
                        "timeline": "investment timeline",
                        "source_quote": "exact quote from transcript"
                    }
                ]
            },
            "infrastructure": {
                "compute_investments": [
                    {
                        "type": "infrastructure type",
                        "scale": "deployment scale",
                        "timeline": "deployment timeline",
                        "investment": "cost/investment amount",
                        "efficiency_gains": "performance improvements",
                        "source_quote": "exact quote from transcript"
                    }
                ],
                "capabilities": [
                    {
                        "name": "capability name",
                        "description": "detailed description",
                        "competitive_advantage": "differentiation factors",
                        "development_stage": "current status",
                        "source_quote": "exact quote from transcript"
                    }
                ]
            }
        },
        "market_analysis": {
            "competitive_position": {
                "strengths": ["specific competitive advantages"],
                "challenges": ["specific challenges or threats"],
                "market_share": "percentage or relative position",
                "key_differentiators": ["unique selling points"],
                "customer_metrics": {
                    "total_customers": "number",
                    "growth_rate": "percentage",
                    "retention_rate": "percentage"
                }
            },
            "industry_trends": [
                {
                    "trend": "specific trend",
                    "impact": "business impact",
                    "company_response": "strategic response",
                    "timeline": "relevant timeline",
                    "market_size": "addressable market size"
                }
            ],
            "growth_drivers": [
                {
                    "driver": "growth driver",
                    "impact": "quantified impact",
                    "timeline": "implementation timeline",
                    "investment_required": "required investment",
                    "risks": ["associated risks"]
                }
            ]
        },
        "risk_factors": {
            "operational": [
                {
                    "risk": "risk description",
                    "potential_impact": "quantified impact",
                    "mitigation": "mitigation strategy"
                }
            ],
            "market": [
                {
                    "risk": "risk description",
                    "potential_impact": "quantified impact",
                    "mitigation": "mitigation strategy"
                }
            ],
            "regulatory": [
                {
                    "risk": "risk description",
                    "potential_impact": "quantified impact",
                    "mitigation": "mitigation strategy"
                }
            ]
        },
        "summary_metrics": {
            "total_ai_revenue": "dollar amount",
            "ai_revenue_growth": "percentage with +/- sign",
            "ai_customer_base": "number of customers",
            "investment_commitment": "dollar amount",
            "key_themes": ["major themes discussed"],
            "major_announcements": ["significant announcements"],
            "sentiment_analysis": {
                "overall_tone": "positive|neutral|negative",
                "confidence": "high|medium|low",
                "key_concerns": ["main concerns raised"],
                "key_opportunities": ["main opportunities highlighted"]
            }
        }
    }

    Extraction Guidelines:
    1. Always provide numerical values where available (dollars, percentages, counts)
    2. Use consistent formatting:
    - Dollar amounts: Include "$" and "B" or "M" (e.g., "$1.2B")
    - Percentages: Include "%" and +/- signs (e.g., "+15.5%")
    - Growth rates: Always indicate direction (e.g., "+", "-")
    3. Include specific timelines and dates where mentioned
    4. For each major data point, include the relevant quote from the transcript
    5. Use "N/A" for unavailable data rather than omitting fields
    6. Prioritize quantitative metrics over qualitative statements
    7. Capture both historical performance and forward-looking guidance
    8. Note any significant year-over-year or sequential changes
    9. Include specific product names, partnership details, and strategic initiatives
    10. Highlight any unexpected or notably significant metrics

    Focus Areas:
    - Quantitative financial metrics
    - Specific AI/ML initiatives and their business impact
    - Concrete timelines and deployment stages
    - Strategic partnerships and their terms
    - Investment amounts and expected returns
    - Market positioning and competitive analysis
    - Risk factors and mitigation strategies
    """,
            "competition_analysis": """
    {Additional prompt for competitive analysis...}
    """,
            "financial_metrics": """
    {Additional prompt for detailed financial analysis...}
    """
    }

        return prompts.get(analysis_type, 'Analyze the transcript for key insights.') + f"\n\nTranscript:\n{transcript}"

    def analyze_transcript(self, transcript: str, analysis_type: str = "earnings_analysis") -> Dict:
        """Analyze transcript content with enhanced parsing."""
        if not transcript:
            return {"error": "No transcript content provided"}

        try:
            response = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": self.generate_prompt(transcript, analysis_type)
                    }
                ]
            )

            cleaned_response = self._clean_response(response.content[0].text)
            
            if cleaned_response.startswith('{'):
                result = json.loads(cleaned_response)
                # Add timestamp to results
                result['analysis_metadata'] = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis_type': analysis_type
                }
                return result
            else:
                return {"error": "Invalid JSON response"}

        except Exception as e:
            print(f"\nError in analysis: {str(e)}")
            return {"error": str(e)}

    def _clean_response(self, response_text: str) -> str:
        """Clean the response text to extract valid JSON."""
        cleaned_response = response_text
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:-3]
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:-3]
        return cleaned_response.strip()

    def generate_summary(self, result: Dict) -> str:
        """Generate a comprehensive summary of the analysis."""
        if isinstance(result, dict) and not result.get('error'):
            summary = []
            
            # Financial Overview
            if fin := result.get('financial_overview', {}):
                summary.append("\n📊 Financial Highlights")
                if metrics := fin.get('key_metrics', {}):
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            summary.append(f"• {key.replace('_', ' ').title()}: {value.get('amount')} ({value.get('growth')})")

            # AI Initiatives
            if ai := result.get('ai_initiatives', {}):
                summary.append("\n🤖 AI Initiatives")
                if products := ai.get('products_and_features', {}):
                    if current := products.get('current'):
                        summary.append("Current Products:")
                        for product in current:
                            summary.append(f"• {product['name']}: {product['description']}")
                
                if partnerships := ai.get('partnerships', {}).get('current'):
                    summary.append("\nKey Partnerships:")
                    for partner in partnerships:
                        summary.append(f"• {partner['partner']}: {partner['description']}")

            # Market Analysis
            if market := result.get('market_analysis', {}):
                summary.append("\n📈 Market Position")
                if position := market.get('competitive_position', {}):
                    if strengths := position.get('strengths'):
                        summary.append("Strengths:")
                        for strength in strengths:
                            summary.append(f"• {strength}")

            # Summary Metrics
            if metrics := result.get('summary_metrics', {}):
                summary.append("\n📌 Key Takeaways")
                if themes := metrics.get('key_themes'):
                    summary.append("Themes: " + ", ".join(themes))
                if revenue := metrics.get('total_ai_revenue'):
                    summary.append(f"AI Revenue: {revenue}")

            return "\n".join(summary)
        else:
            return "\nError: Could not generate summary due to invalid data structure."