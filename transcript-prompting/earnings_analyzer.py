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
Analyze the provided earnings call transcript in detail, focusing on the following areas:

1. **AI Initiatives**:
- Identify and describe all AI-related projects, strategies, and investments.
- Highlight the stage of each initiative (e.g., announced, in development, launched).
- Provide timelines, investment amounts, and expected business impacts.

2. **Financial Metrics**:
- Extract key financial figures such as revenue, operating income, EPS, and cash flow.
- Include growth rates (year-over-year and sequential), margins, and comparisons to consensus estimates.
- Reference exact quotes from the transcript that support these metrics.

3. **Strategic Insights**:
- Summarize strategic moves, including partnerships, mergers, acquisitions, and market positioning.
- Detail any guidance provided for future quarters or the full year, including revenue and growth projections.
- Note any risk factors mentioned and their potential impacts.

Return the analysis as a structured JSON object adhering to the following schema:

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
                "operating_margin_range": "expected margin range",
                "key_drivers": ["specific growth drivers"],
                "assumptions": ["key assumptions"],
                "source_quote": "exact quote from transcript"
            },
            "full_year": {
                "revenue_range": "expected range in dollars",
                "growth_range": "expected growth range",
                "operating_margin_range": "expected margin range",
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

**Extraction Guidelines:**
1. **Numerical Precision**:
- Always provide numerical values where available (e.g., dollars, percentages, counts).
- Use consistent formatting:
    - **Dollar amounts**: Include "$" and abbreviations like "B" for billions or "M" for millions (e.g., "$1.2B").
    - **Percentages**: Include "%" and indicate direction with "+" or "-" signs (e.g., "+15.5%").
    - **Growth rates**: Always indicate the direction (e.g., "+", "-").

2. **Quotations**:
- For each major data point, include the relevant exact quote from the transcript to support the information.

3. **Completeness**:
- Use "N/A" for unavailable data instead of omitting fields.
- Capture both historical performance and forward-looking guidance.
- Note any significant year-over-year or sequential changes.

4. **Specificity**:
- Include specific product names, partnership details, and strategic initiatives.
- Provide concrete timelines and deployment stages where mentioned.

5. **Prioritization**:
- Prioritize quantitative metrics over qualitative statements.
- Highlight any unexpected or notably significant metrics.

6. **Focus Areas**:
- **Quantitative Financial Metrics**: Detailed extraction of all key financial figures.
- **AI/ML Initiatives**: Specific initiatives and their business impacts.
- **Strategic Partnerships**: Terms and strategic value of partnerships.
- **Investment Details**: Amounts and expected returns from investments.
- **Market Positioning**: Competitive analysis and market share details.
- **Risk Factors**: Identification and mitigation strategies for various risks.

Ensure that the final JSON strictly adheres to the provided schema for consistency and ease of further processing.

**Important**: **Only output the JSON object. Do not include any explanations, comments, or additional text.**
""",
            "competition_analysis": """
Conduct a comprehensive competitive analysis based on the provided earnings call transcript. Focus on the following areas:

1. **Competitive Landscape**:
- Identify key competitors mentioned in the transcript.
- Describe the market position of each competitor relative to the company.

2. **Strengths and Weaknesses**:
- Highlight the company's strengths compared to competitors.
- Detail any weaknesses or areas where competitors have an advantage.

3. **Market Share and Positioning**:
- Provide data on market share percentages if available.
- Discuss how the company differentiates itself in the market.

4. **Competitive Strategies**:
- Summarize strategies the company is employing to gain a competitive edge.
- Include any partnerships, innovations, or strategic initiatives aimed at outperforming competitors.

5. **Challenges and Threats**:
- Identify any competitive threats or challenges mentioned.
- Explain how the company plans to address these threats.

6. **Customer Perception**:
- Include insights into customer satisfaction, loyalty, and preferences relative to competitors.

Return the analysis as a structured JSON object following this schema:

{
    "competitive_landscape": {
        "key_competitors": ["competitor1", "competitor2", ...],
        "company_position": "description of company's market position",
        "market_share": {
            "company": "percentage",
            "competitors": {
                "competitor1": "percentage",
                "competitor2": "percentage",
                ...
            }
        }
    },
    "strengths_weaknesses": {
        "strengths": ["strength1", "strength2", ...],
        "weaknesses": ["weakness1", "weakness2", ...]
    },
    "competitive_strategies": {
        "current_strategies": ["strategy1", "strategy2", ...],
        "future_plans": ["plan1", "plan2", ...]
    },
    "challenges_threats": {
        "challenges": ["challenge1", "challenge2", ...],
        "mitigation_plans": ["plan1", "plan2", ...]
    },
    "customer_perception": {
        "satisfaction": "description or percentage",
        "loyalty": "description or percentage",
        "preferences": ["preference1", "preference2", ...]
    }
}

**Extraction Guidelines:**
1. **Accuracy**:
- Ensure all competitor names and related data are accurately extracted.
- Use exact quotes from the transcript to support each point.

2. **Quantitative Data**:
- Provide numerical data where available, such as market share percentages.

3. **Clarity and Structure**:
- Organize the analysis clearly under each schema category.
- Use bullet points or lists within the JSON structure for readability.

4. **Completeness**:
- Do not omit any section; use "N/A" if certain information is not available.

5. **Specificity**:
- Provide specific examples or instances where possible to illustrate points.

6. **Focus Areas**:
- Emphasize the company's competitive advantages and areas needing improvement.
- Highlight strategic actions taken to enhance competitiveness.

Ensure the final JSON adheres strictly to the schema for consistency and ease of processing.

**Important**: **Only output the JSON object. Do not include any explanations, comments, or additional text.**
""",
            "financial_metrics": """
Perform an in-depth financial analysis based on the provided earnings call transcript. Focus on the following areas:

1. **Income Statement Analysis**:
- Extract detailed information on revenue, cost of goods sold (COGS), gross profit, operating expenses, operating income, net income, and EPS.
- Provide growth rates (year-over-year and sequential) for each line item.

2. **Balance Sheet Metrics**:
- Summarize key balance sheet items such as total assets, liabilities, and shareholders' equity.
- Highlight changes in these metrics compared to previous periods.

3. **Cash Flow Analysis**:
- Detail cash flows from operating, investing, and financing activities.
- Highlight free cash flow and any significant cash movements.

4. **Profitability Ratios**:
- Calculate and provide ratios such as gross margin, operating margin, net margin, return on assets (ROA), and return on equity (ROE).

5. **Liquidity and Solvency Ratios**:
- Provide ratios like current ratio, quick ratio, debt-to-equity ratio, and interest coverage ratio.

6. **Efficiency Ratios**:
- Include metrics such as inventory turnover, receivables turnover, and asset turnover.

7. **Earnings Guidance**:
- Summarize any guidance provided for future earnings, including revenue and EPS projections.
- Include the range and any factors influencing the guidance.

8. **Comparative Analysis**:
- Compare current financial performance with previous periods and industry benchmarks if available.

Return the analysis as a structured JSON object following this schema:

{
    "income_statement": {
        "revenue": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "year_over_year_growth": "percentage with +/- sign",
            "sequential_growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "cogs": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "gross_profit": {
            "current_period": "dollar amount",
            "margin": "percentage",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "operating_expenses": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "operating_income": {
            "current_period": "dollar amount",
            "margin": "percentage",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "net_income": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "eps": {
            "current_period": "dollar amount per share",
            "previous_period": "dollar amount per share",
            "growth": "percentage with +/- sign",
            "beat_miss": "difference from consensus",
            "source_quote": "exact quote from transcript"
        }
    },
    "balance_sheet": {
        "total_assets": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "total_liabilities": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "shareholders_equity": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        }
    },
    "cash_flow": {
        "operating_cash_flow": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "investing_cash_flow": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "financing_cash_flow": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "free_cash_flow": {
            "current_period": "dollar amount",
            "previous_period": "dollar amount",
            "growth": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        }
    },
    "profitability_ratios": {
        "gross_margin": "percentage",
        "operating_margin": "percentage",
        "net_margin": "percentage",
        "return_on_assets": "percentage",
        "return_on_equity": "percentage",
        "source_quote": "exact quote from transcript"
    },
    "liquidity_solveny_ratios": {
        "current_ratio": "ratio",
        "quick_ratio": "ratio",
        "debt_to_equity_ratio": "ratio",
        "interest_coverage_ratio": "ratio",
        "source_quote": "exact quote from transcript"
    },
    "efficiency_ratios": {
        "inventory_turnover": "times",
        "receivables_turnover": "times",
        "asset_turnover": "times",
        "source_quote": "exact quote from transcript"
    },
    "earnings_guidance": {
        "revenue_projection": {
            "range": "dollar amount range",
            "growth_expectation": "percentage with +/- sign",
            "key_factors": ["factor1", "factor2", ...],
            "source_quote": "exact quote from transcript"
        },
        "eps_projection": {
            "range": "dollar amount per share range",
            "growth_expectation": "percentage with +/- sign",
            "key_factors": ["factor1", "factor2", ...],
            "source_quote": "exact quote from transcript"
        }
    },
    "comparative_analysis": {
        "current_vs_previous": {
            "revenue_change": "percentage with +/- sign",
            "net_income_change": "percentage with +/- sign",
            "eps_change": "percentage with +/- sign",
            "source_quote": "exact quote from transcript"
        },
        "industry_benchmarks": {
            "revenue_growth": "company vs industry",
            "profit_margin": "company vs industry",
            "other_key_metrics": "comparison details",
            "source_quote": "exact quote from transcript"
        }
    }
}

**Extraction Guidelines:**
1. **Numerical Accuracy**:
- Provide precise numerical values for all financial metrics.
- Ensure consistency in formatting (e.g., "$1.2B", "+15.5%").

2. **Comprehensive Coverage**:
- Extract all relevant financial data mentioned in the transcript.
- Do not omit any section; use "N/A" if certain information is unavailable.

3. **Supporting Evidence**:
- Include exact quotes from the transcript to back each extracted data point.

4. **Ratio Calculations**:
- Where ratios are not explicitly mentioned, calculate them using the provided data if possible.
- If not enough data is available, mark the ratio as "N/A".

5. **Clarity and Structure**:
- Organize the JSON strictly according to the provided schema.
- Use clear and descriptive keys to ensure ease of understanding and processing.

6. **Comparative Insights**:
- Highlight significant changes compared to previous periods.
- Compare company performance against industry benchmarks when available.

7. **Focus Areas**:
- Emphasize key financial strengths and areas of concern.
- Highlight any guidance provided and the factors influencing future performance.

Ensure that the final JSON strictly adheres to the schema for consistency and ease of further processing.

**Important**: **Only output the JSON object. Do not include any explanations, comments, or additional text.**
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
                            summary.append(
                                f"• {key.replace('_', ' ').title()}: {value.get('amount')} ({value.get('growth')})")

            # AI Initiatives
            if ai := result.get('ai_initiatives', {}):
                summary.append("\n🤖 AI Initiatives")
                if products := ai.get('products_and_features', {}):
                    if current := products.get('current'):
                        summary.append("Current Products:")
                        for product in current:
                            summary.append(
                                f"• {product['name']}: {product['description']}")

                if partnerships := ai.get('partnerships', {}).get('current'):
                    summary.append("\nKey Partnerships:")
                    for partner in partnerships:
                        summary.append(
                            f"• {partner['partner']}: {partner['description']}")

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
