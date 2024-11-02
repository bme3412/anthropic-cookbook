# utils.py
import json
from datetime import datetime
from typing import Dict, Any
from models import EarningsAnalysis

def save_analysis_to_json(analysis: EarningsAnalysis, usage_stats: dict, filename: str) -> str:
    """Save analysis results to JSON file"""
    try:
        analysis_dict = {
            "company": analysis.company,
            "date": analysis.date.isoformat(),
            "quarter": (analysis.date.month + 2) // 3,
            "year": analysis.date.year,
            "summary": analysis.summary,
            "key_highlights": analysis.key_highlights,
            "metrics": [metric.dict() for metric in analysis.metrics],
            "kpis": [kpi.dict() for kpi in analysis.kpis],
            "operational_metrics": [metric.dict() for metric in analysis.operational_metrics],
            "risk_factors": [risk.dict() for risk in analysis.risk_factors],
            "strategic_initiatives": [initiative.dict() for initiative in analysis.strategic_initiatives],
            "usage_statistics": usage_stats
        }

        with open(filename, 'w') as f:
            json.dump(analysis_dict, f, indent=2)

        return filename
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")
        return None

def load_analysis_from_json(filename: str) -> Dict[str, Any]:
    """Load analysis from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading analysis: {str(e)}")
        return None

class PromptGenerator:
    """Generate analysis prompts for different company transcripts with enhanced financial metrics focus"""
    
    @staticmethod
    def get_financial_metrics_prompt(company: str) -> str:
        return f"""
        Extract all financial metrics mentioned in the earnings call transcript for {company}. 
        For each metric, provide detailed analysis across the following categories:

        1. Revenue Metrics:
           - Total revenue
           - Revenue by segment/product line
           - Revenue growth rates (YoY, QoQ)
           - Geographic revenue breakdown
           - Recurring revenue metrics

        2. Profitability Metrics:
           - Gross profit and margin
           - Operating income and margin
           - EBITDA and margin
           - Net income and margin
           - Earnings per share (Basic and Diluted)

        3. Cash Flow Metrics:
           - Operating cash flow
           - Free cash flow
           - Capital expenditure
           - Cash conversion rate

        4. Balance Sheet Metrics:
           - Cash and equivalents
           - Total assets
           - Total debt
           - Equity
           - Working capital

        5. Operational Financial Metrics:
           - Days sales outstanding
           - Inventory turnover
           - Operating expenses
           - R&D expenses
           - Marketing expenses

        For each metric, provide:
        1. Metric name
        2. Numerical value
        3. Time period (quarter or annual)
        4. Year
        5. Quarter number (if applicable)
        6. YoY change (%)
        7. QoQ change (%)
        8. Description or context

        Return the data in the following JSON format:
        {{
            "financial_metrics": [
                {{
                    "category": "Revenue",
                    "metric_name": "Total Revenue",
                    "value": 50000.0,
                    "period": "Quarterly",
                    "year": 2023,
                    "quarter": 4,
                    "yoy_change": 15.5,
                    "qoq_change": 3.2,
                    "description": "Total consolidated revenue across all segments"
                }}
            ]
        }}
        """

    @staticmethod
    def get_kpis_prompt(company: str) -> str:
        return f"""
        Extract key performance indicators for {company} focusing on:
        1. Core business metrics
           - Market share
           - Customer acquisition cost
           - Customer lifetime value
           - Unit economics
        2. Product/service performance
           - Product mix
           - Service adoption rates
           - Cross-sell/upsell metrics
        3. Customer/user metrics
           - Active users/customers
           - Churn rate
           - Net promoter score
           - Customer satisfaction
        4. Growth metrics
           - New market expansion
           - Product launch metrics
           - Customer base growth
        5. Efficiency metrics
           - Resource utilization
           - Productivity metrics
           - Cost per unit

        Return the data in the following JSON format:
        {{
            "kpis": [
                {{
                    "name": "Market Share",
                    "value": "32% in primary market",
                    "context": "Increased from 28% in previous quarter"
                }}
            ]
        }}
        """

    @staticmethod
    def get_operational_metrics_prompt(company: str) -> str:
        return f"""
        Extract detailed operational metrics for {company} focusing on:
        1. Production/Service Delivery
           - Production volume
           - Capacity utilization
           - Order fulfillment rate
           - Service level metrics
        2. Efficiency Metrics
           - Production costs
           - Labor productivity
           - Asset utilization
           - Process efficiency
        3. Quality Metrics
           - Defect rates
           - Customer satisfaction
           - Service reliability
           - Quality scores
        4. Time-Based Metrics
           - Cycle time
           - Lead time
           - Response time
           - Time to market

        Return the data in the following JSON format:
        {{
            "operational_metrics": [
                {{
                    "metric_name": "Production Efficiency",
                    "value": "95.5%",
                    "description": "Manufacturing capacity utilization rate"
                }}
            ]
        }}
        """

    @staticmethod
    def get_strategic_initiatives_prompt(company: str) -> str:
        return f"""
        Extract strategic initiatives mentioned in the earnings call transcript for {company}. 
        Focus on:
        1. Growth Initiatives
           - Market expansion
           - Product development
           - M&A activities
        2. Operational Excellence
           - Cost optimization
           - Process improvements
           - Technology upgrades
        3. Innovation
           - R&D projects
           - Digital transformation
           - New technologies
        4. Market Position
           - Competitive strategies
           - Brand initiatives
           - Partnership developments

        Return the data in the following JSON format:
        {{
            "strategic_initiatives": [
                {{
                    "initiative_name": "Digital Transformation",
                    "progress": "Completed cloud migration of core systems",
                    "impact": "30% improvement in processing speed",
                    "timeline": "Q4 2023 completion"
                }}
            ]
        }}
        """

    @staticmethod
    def get_summary_prompt(company: str) -> str:
        return f"""
        Provide a comprehensive financial and operational analysis of {company}'s earnings call in the following JSON format:
        {{
            "summary": "string containing detailed analysis",
            "highlights": ["string1", "string2", "string3", "string4", "string5"],
            "financial_outlook": {{
                "short_term": "string containing next quarter outlook",
                "long_term": "string containing annual/multi-year outlook",
                "key_drivers": ["driver1", "driver2", "driver3"]
            }}
        }}

        The summary should focus on:
        1. Financial performance across all key metrics
        2. Operational achievements and efficiency gains
        3. Strategic initiatives and their financial impact
        4. Market positioning and competitive advantages
        5. Forward-looking guidance and growth expectations

        The highlights should be the 5 most impactful financial and operational takeaways from the call.
        """