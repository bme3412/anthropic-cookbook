import argparse
from typing import Dict, List, Optional, Tuple
import os
from pydantic import BaseModel, Field
from datetime import datetime
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import json
from openai import OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# ---------------------------
# Data Models
# ---------------------------

class TokenUsageTracker:
    """Track token usage and costs across different models"""

    # Current OpenAI pricing per 1k tokens (as of March 2024)
    PRICE_PER_1K_TOKENS = {
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},  # Cheapest GPT-4 variant
        'text-embedding-3-small': {'input': 0.00002, 'output': 0.00002}  # Cheapest embedding model
    }

    def __init__(self):
        self.usage = {
            'gpt-4o-mini': {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0},
            'text-embedding-3-small': {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
        }

        # Initialize tokenizers
        try:
            self.encoders = {
                'gpt-4o-mini': tiktoken.get_encoding("cl100k_base"),
                'text-embedding-3-small': tiktoken.get_encoding("cl100k_base")
            }
        except Exception as e:
            print(f"Error initializing tokenizers: {e}")
            # Fallback to p50k_base if cl100k_base is not available
            self.encoders = {
                'gpt-4o-mini': tiktoken.get_encoding("p50k_base"),
                'text-embedding-3-small': tiktoken.get_encoding("p50k_base")
            }

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for a given text and model"""
        try:
            return len(self.encoders[model].encode(text))
        except Exception as e:
            print(f"Error counting tokens for {model}: {e}")
            return 0  # Return 0 on error to allow continued execution

    def add_usage(self, model: str, input_tokens: int, output_tokens: int = 0):
        """Add token usage and calculate cost"""
        if model not in self.usage:
            print(f"Warning: Unknown model {model}")
            return

        self.usage[model]['input_tokens'] += input_tokens
        self.usage[model]['output_tokens'] += output_tokens

        # Calculate cost
        input_cost = (input_tokens / 1000) * self.PRICE_PER_1K_TOKENS[model]['input']
        output_cost = (output_tokens / 1000) * self.PRICE_PER_1K_TOKENS[model]['output']
        self.usage[model]['cost'] += input_cost + output_cost

    def get_total_usage(self) -> dict:
        """Get total token usage and cost across all models"""
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


class OpenAIQuotaError(Exception):
    """Custom exception for OpenAI quota errors"""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class APIError(Exception):
    """Custom exception for API-related errors"""
    pass


class FinancialMetric(BaseModel):
    metric_name: str
    value: float
    period: str
    year: int
    quarter: Optional[int]


class KPI(BaseModel):
    name: str
    value: str
    context: Optional[str]


class OperationalMetric(BaseModel):
    metric_name: str
    value: float
    description: Optional[str]


class RiskFactor(BaseModel):
    risk_name: str
    description: str


class StrategicInitiative(BaseModel):
    initiative_name: str
    progress: str
    impact: Optional[str]


class EarningsAnalysis(BaseModel):
    company: str = "MSFT"
    date: datetime
    metrics: List[FinancialMetric] = []
    kpis: List[KPI] = []
    operational_metrics: List[OperationalMetric] = []
    risk_factors: List[RiskFactor] = []
    strategic_initiatives: List[StrategicInitiative] = []
    summary: str
    key_highlights: List[str]


# ---------------------------
# Transcript Analyzer Class
# ---------------------------

class TranscriptAnalyzer:
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """Initialize analyzer with environment variables and retry settings"""
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize token tracker
        self.token_tracker = TokenUsageTracker()

        if not self.fmp_api_key:
            raise ConfigurationError("FMP_API_KEY not found in environment variables")
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY not found in environment variables")

        try:
            # Initialize LLM components with optimal models
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",  # Using cheapest embedding model
                openai_api_key=self.openai_api_key,
                timeout=60,
                max_retries=self.max_retries
            )
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4o-mini",  # Using cost-efficient GPT-4 variant
                openai_api_key=self.openai_api_key,
                timeout=60,
                max_retries=self.max_retries
            )
        except OpenAIError as e:
            if "insufficient_quota" in str(e):
                raise OpenAIQuotaError("OpenAI API quota exceeded. Please check your billing details.") from e
            raise APIError(f"Error initializing OpenAI components: {str(e)}") from e

        # Configure analysis settings from environment variables
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.model_temperature = float(os.getenv('MODEL_TEMPERATURE', '0'))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda _: None
    )
    def get_transcript(self, symbol: str, quarter: int, year: int) -> str:
        """Fetch earnings call transcript with improved error handling"""
        base_url = os.getenv('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')
        url = f"{base_url}/earning_call_transcript/{symbol}"
        params = {
            'quarter': quarter,
            'year': year,
            'apikey': self.fmp_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                raise APIError("No transcript data received")

            transcript = data[0].get("content")
            if not transcript:
                raise APIError("Transcript content is empty")

            # Clean up the transcript
            transcript = transcript.replace('\n', ' ').replace('\r', ' ')
            return transcript

        except requests.exceptions.RequestException as e:
            raise APIError(f"Failed to fetch transcript: {str(e)}")
        except (IndexError, KeyError) as e:
            raise APIError(f"Invalid transcript data structure: {str(e)}")

    def create_vector_store(self, transcript: str) -> FAISS:
        """Create vector store with token tracking"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_text(transcript)
            print(f"Number of chunks created: {len(chunks)}")  # Debug statement

            # Track embedding tokens
            total_embedding_tokens = sum(
                self.token_tracker.count_tokens(chunk, 'text-embedding-3-small')
                for chunk in chunks
            )
            self.token_tracker.add_usage('text-embedding-3-small', total_embedding_tokens)

            return FAISS.from_texts(chunks, self.embeddings)
        except Exception as e:
            raise APIError(f"Failed to create vector store: {str(e)}")

    def safe_parse_json(self, response: str | dict) -> dict:
        """Safely parse JSON response with improved handling"""
        try:
            if isinstance(response, dict):
                # If response is a dict, return it directly
                return response
            if isinstance(response, str):
                return json.loads(response)
            raise ValueError(f"Unexpected response type: {type(response)}")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e} - Response: {response}")  # Improved logging
            try:
                if isinstance(response, str):
                    # Remove any potential markdown formatting
                    clean_response = response.replace('```json', '').replace('```', '')
                    return json.loads(clean_response)
                raise e
            except Exception as inner_e:
                if isinstance(response, str):
                    if any(bad_word in response.lower() for bad_word in ['import', 'exec', 'eval', 'os.', 'system']):
                        raise ValueError("Potentially unsafe content in response")
                    try:
                        # Attempt safe evaluation
                        return eval(clean_response, {"__builtins__": {}})
                    except Exception as eval_e:
                        print(f"Eval failed: {eval_e}")
                raise ValueError(f"Could not parse response of type {type(response)}")

    def extract_financial_metrics(self, vector_store: FAISS) -> List[FinancialMetric]:
        """Extract detailed financial metrics"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}  # Increased k for more context
                )
            )

            financial_metrics_prompt = os.getenv('FINANCIAL_METRICS_PROMPT', """
            Extract all financial metrics mentioned in the context. For each metric, provide:
            1. Metric name (e.g., "Revenue", "Operating Income", "EPS")
            2. Numerical value
            3. Time period (quarter or annual)
            4. Year
            5. Quarter number (if applicable)
            6. Description (optional)

            Return the data in the following JSON format:
            {
                "financial_metrics": [
                    {
                        "metric_name": "Revenue",
                        "value": 50000.0,
                        "period": "Quarterly",
                        "year": 2023,
                        "quarter": 4,
                        "description": "Total revenue from all business segments."
                    }
                ]
            }

            Ensure that the JSON is properly formatted and free of errors.
            """)

            # Track input tokens
            input_tokens = self.token_tracker.count_tokens(financial_metrics_prompt, 'gpt-4o-mini')

            # Get response
            response = qa_chain.invoke(financial_metrics_prompt)
            # Extract 'result' from the response
            llm_result = response.get('result', '')
            print(f"Financial Metrics Response: {llm_result}")  # Debug statement

            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse response and validate
            financial_metrics_data = self.safe_parse_json(llm_result)
            if not isinstance(financial_metrics_data, dict) or 'financial_metrics' not in financial_metrics_data:
                print("Financial Metrics data is not in expected format.")
                return []

            return [FinancialMetric(**metric) for metric in financial_metrics_data['financial_metrics']]

        except Exception as e:
            print(f"Error extracting financial metrics: {str(e)}")
            return []

    def extract_kpis(self, vector_store: FAISS) -> List[KPI]:
        """Extract KPIs with improved response handling"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}  # Increased k for more context
                )
            )

            kpi_prompt = os.getenv('KPI_PROMPT', """
            Extract Microsoft's key performance indicators and business metrics. Focus on:
            1. Cloud services metrics (Azure growth, cloud revenue)
            2. Product performance (Office 365, LinkedIn, Gaming)
            3. Customer/user metrics
            4. Growth rates and trends
            5. Operational efficiency metrics

            Return the data in the following JSON format:
            {
                "kpis": [
                    {
                        "name": "Azure Growth",
                        "value": "30% YoY increase",
                        "context": "Azure cloud services saw a 30% year-over-year increase in revenue.",
                        "description": "Growth rate of Azure's revenue compared to the previous year."
                    }
                ]
            }

            Ensure that the JSON is properly formatted and free of errors.
            """)

            # Track input tokens
            input_tokens = self.token_tracker.count_tokens(kpi_prompt, 'gpt-4o-mini')

            # Get response
            response = qa_chain.invoke(kpi_prompt)
            # Extract 'result' from the response
            llm_result = response.get('result', '')
            print(f"KPIs Response: {llm_result}")  # Debug statement

            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse response and validate
            kpi_data = self.safe_parse_json(llm_result)
            if not isinstance(kpi_data, dict) or 'kpis' not in kpi_data:
                print("KPIs data is not in expected format.")
                return []

            return [KPI(**kpi) for kpi in kpi_data['kpis']]

        except Exception as e:
            print(f"Error extracting KPIs: {str(e)}")
            return []

    def extract_operational_metrics(self, vector_store: FAISS) -> List[OperationalMetric]:
        """Extract operational metrics"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}
                )
            )

            operational_metrics_prompt = os.getenv('OPERATIONAL_METRICS_PROMPT', """
            Extract operational metrics mentioned in the context. For each metric, provide:
            1. Metric name (e.g., "Customer Retention Rate", "Operational Efficiency")
            2. Numerical value
            3. Description (optional)

            Return the data in the following JSON format:
            {
                "operational_metrics": [
                    {
                        "metric_name": "Customer Retention Rate",
                        "value": 85.0,
                        "description": "Percentage of customers retained compared to the previous year."
                    }
                ]
            }

            Ensure that the JSON is properly formatted and free of errors.
            """)

            # Track input tokens
            input_tokens = self.token_tracker.count_tokens(operational_metrics_prompt, 'gpt-4o-mini')

            # Get response
            response = qa_chain.invoke(operational_metrics_prompt)
            # Extract 'result' from the response
            llm_result = response.get('result', '')
            print(f"Operational Metrics Response: {llm_result}")  # Debug statement

            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse response and validate
            operational_metrics_data = self.safe_parse_json(llm_result)
            if not isinstance(operational_metrics_data, dict) or 'operational_metrics' not in operational_metrics_data:
                print("Operational Metrics data is not in expected format.")
                return []

            return [OperationalMetric(**metric) for metric in operational_metrics_data['operational_metrics']]

        except Exception as e:
            print(f"Error extracting operational metrics: {str(e)}")
            return []

    def extract_risk_factors(self, vector_store: FAISS) -> List[RiskFactor]:
        """Extract risk factors mentioned in the transcript"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}
                )
            )

            risk_factors_prompt = os.getenv('RISK_FACTORS_PROMPT', """
            Extract risk factors mentioned in the context. For each risk factor, provide:
            1. Risk name
            2. Description

            Return the data in the following JSON format:
            {
                "risk_factors": [
                    {
                        "risk_name": "Market Volatility",
                        "description": "Fluctuations in market conditions could adversely affect revenue and profitability."
                    }
                ]
            }

            Ensure that the JSON is properly formatted and free of errors.
            """)

            # Track input tokens
            input_tokens = self.token_tracker.count_tokens(risk_factors_prompt, 'gpt-4o-mini')

            # Get response
            response = qa_chain.invoke(risk_factors_prompt)
            # Extract 'result' from the response
            llm_result = response.get('result', '')
            print(f"Risk Factors Response: {llm_result}")  # Debug statement

            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse response and validate
            risk_factors_data = self.safe_parse_json(llm_result)
            if not isinstance(risk_factors_data, dict) or 'risk_factors' not in risk_factors_data:
                print("Risk Factors data is not in expected format.")
                return []

            return [RiskFactor(**risk) for risk in risk_factors_data['risk_factors']]

        except Exception as e:
            print(f"Error extracting risk factors: {str(e)}")
            return []

    def extract_strategic_initiatives(self, vector_store: FAISS) -> List[StrategicInitiative]:
        """Extract strategic initiatives mentioned in the transcript"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}
                )
            )

            strategic_initiatives_prompt = os.getenv('STRATEGIC_INITIATIVES_PROMPT', """
            Extract strategic initiatives mentioned in the context. For each initiative, provide:
            1. Initiative name
            2. Progress update
            3. Impact (optional)

            Return the data in the following JSON format:
            {
                "strategic_initiatives": [
                    {
                        "initiative_name": "AI Integration",
                        "progress": "Completed integration of AI into Office 365.",
                        "impact": "Enhanced user productivity by 15%."
                    }
                ]
            }

            Ensure that the JSON is properly formatted and free of errors.
            """)

            # Track input tokens
            input_tokens = self.token_tracker.count_tokens(strategic_initiatives_prompt, 'gpt-4o-mini')

            # Get response
            response = qa_chain.invoke(strategic_initiatives_prompt)
            # Extract 'result' from the response
            llm_result = response.get('result', '')
            print(f"Strategic Initiatives Response: {llm_result}")  # Debug statement

            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse response and validate
            strategic_initiatives_data = self.safe_parse_json(llm_result)
            if not isinstance(strategic_initiatives_data, dict) or 'strategic_initiatives' not in strategic_initiatives_data:
                print("Strategic Initiatives data is not in expected format.")
                return []

            return [StrategicInitiative(**initiative) for initiative in strategic_initiatives_data['strategic_initiatives']]

        except Exception as e:
            print(f"Error extracting strategic initiatives: {str(e)}")
            return []

    def generate_summary(self, vector_store: FAISS) -> Tuple[str, List[str]]:
        """Generate summary with improved response handling"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}
                )
            )

            summary_prompt = os.getenv('SUMMARY_PROMPT', """
            Provide a detailed analysis of Microsoft's earnings call in the following JSON format:
            {
                "summary": "string containing comprehensive analysis",
                "highlights": ["string1", "string2", "string3", "string4", "string5"]
            }

            The summary should cover:
            1. Overall financial performance and key metrics
            2. Strategic initiatives and their progress
            3. Market conditions and future outlook
            4. Notable product or service updates
            5. Any significant challenges or risks mentioned

            The highlights should be the 5 most important takeaways from the call.

            Ensure that the JSON is properly formatted and free of errors.
            """)

            # Track input tokens
            input_tokens = self.token_tracker.count_tokens(summary_prompt, 'gpt-4o-mini')

            # Get response
            response = qa_chain.invoke(summary_prompt)
            # Extract 'result' from the response
            llm_result = response.get('result', '')
            print(f"Summary Response: {llm_result}")  # Debug statement

            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse response and validate
            summary_data = self.safe_parse_json(llm_result)
            if not isinstance(summary_data, dict):
                print("Summary data is not in expected format.")
                return "", []

            return summary_data.get('summary', ""), summary_data.get('highlights', [])

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "", []

    def analyze_transcript(self, symbol: str, quarter: int, year: int) -> Tuple[EarningsAnalysis, dict]:
        """Main analysis function returning both analysis and usage statistics"""
        try:
            transcript = self.get_transcript(symbol, quarter, year)
            print(f"Transcript fetched: {transcript[:500]}...")  # Print the first 500 characters for debugging
            vector_store = self.create_vector_store(transcript)

            metrics = self.extract_financial_metrics(vector_store)
            kpis = self.extract_kpis(vector_store)
            operational_metrics = self.extract_operational_metrics(vector_store)
            risk_factors = self.extract_risk_factors(vector_store)
            strategic_initiatives = self.extract_strategic_initiatives(vector_store)
            summary, highlights = self.generate_summary(vector_store)

            actual_quarter = ((quarter - 1) % 4) + 1

            analysis = EarningsAnalysis(
                company=symbol,
                date=datetime(year, quarter * 3, 1),
                metrics=metrics,
                kpis=kpis,
                operational_metrics=operational_metrics,
                risk_factors=risk_factors,
                strategic_initiatives=strategic_initiatives,
                summary=summary,
                key_highlights=highlights
            )

            # Get usage statistics
            usage_stats = self.token_tracker.get_total_usage()

            return analysis, usage_stats

        except Exception as e:
            raise APIError(f"Analysis failed: {str(e)}")


# ---------------------------
# Utility Functions
# ---------------------------

def save_analysis_to_json(analysis: EarningsAnalysis, usage_stats: dict, filename: str = "msft_analysis.json"):
    """Save analysis results to JSON file"""
    try:
        # Convert analysis to dict format
        analysis_dict = {
            "company": analysis.company,
            "date": analysis.date.isoformat(),
            "quarter": (analysis.date.month + 2) // 3,
            "year": analysis.date.year,
            "summary": analysis.summary,
            "key_highlights": analysis.key_highlights,
            "metrics": [
                {
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "period": metric.period,
                    "year": metric.year,
                    "quarter": metric.quarter
                }
                for metric in analysis.metrics
            ],
            "kpis": [
                {
                    "name": kpi.name,
                    "value": kpi.value,
                    "context": kpi.context
                }
                for kpi in analysis.kpis
            ],
            "operational_metrics": [
                {
                    "metric_name": op_metric.metric_name,
                    "value": op_metric.value,
                    "description": op_metric.description
                }
                for op_metric in analysis.operational_metrics
            ],
            "risk_factors": [
                {
                    "risk_name": risk.risk_name,
                    "description": risk.description
                }
                for risk in analysis.risk_factors
            ],
            "strategic_initiatives": [
                {
                    "initiative_name": initiative.initiative_name,
                    "progress": initiative.progress,
                    "impact": initiative.impact
                }
                for initiative in analysis.strategic_initiatives
            ],
            "usage_statistics": usage_stats
        }

        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(analysis_dict, f, indent=2)

        return filename
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")
        return None


def load_and_display_analysis(filename: str = "msft_analysis.json"):
    """Load and display analysis from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        print(f"\nAnalysis for {data['company']} Q{data['quarter']} {data['year']}")
        print("\nSummary:")
        print(data['summary'])

        print("\nKey Highlights:")
        for highlight in data['key_highlights']:
            print(f"- {highlight}")

        print("\nFinancial Metrics:")
        for metric in data['metrics']:
            print(f"- {metric['metric_name']}: {metric['value']} ({metric['period']})")

        print("\nKPIs:")
        for kpi in data['kpis']:
            print(f"- {kpi['name']}: {kpi['value']}")
            if kpi['context']:
                print(f"  Context: {kpi['context']}")

        print("\nOperational Metrics:")
        for op_metric in data['operational_metrics']:
            print(f"- {op_metric['metric_name']}: {op_metric['value']}")
            if op_metric['description']:
                print(f"  Description: {op_metric['description']}")

        print("\nRisk Factors:")
        for risk in data['risk_factors']:
            print(f"- {risk['risk_name']}: {risk['description']}")

        print("\nStrategic Initiatives:")
        for initiative in data['strategic_initiatives']:
            print(f"- {initiative['initiative_name']}: {initiative['progress']}")
            if initiative['impact']:
                print(f"  Impact: {initiative['impact']}")

        print("\nToken Usage and Cost Summary:")
        usage = data['usage_statistics']
        print(f"Total Tokens Used: {usage['total_tokens']:,}")
        print(f"Total Cost: ${usage['total_cost']:.4f}")

        print("\nDetailed Usage by Model:")
        for model, stats in usage['detailed_usage'].items():
            print(f"\n{model}:")
            print(f"  Input Tokens: {stats['input_tokens']:,}")
            print(f"  Output Tokens: {stats['output_tokens']:,}")
            print(f"  Cost: ${stats['cost']:.4f}")

    except Exception as e:
        print(f"Error loading analysis: {str(e)}")


# ---------------------------
# Command-Line Interface
# ---------------------------

def cli_main():
    parser = argparse.ArgumentParser(description="Earnings Call Transcript Analyzer")
    parser.add_argument('--symbol', type=str, default='MSFT', help='Stock symbol of the company')
    parser.add_argument('--quarter', type=int, required=True, help='Quarter number (1-4)')
    parser.add_argument('--year', type=int, required=True, help='Fiscal year')

    args = parser.parse_args()

    try:
        analyzer = TranscriptAnalyzer(max_retries=3, retry_delay=5)
        analysis, usage_stats = analyzer.analyze_transcript(symbol=args.symbol, quarter=args.quarter, year=args.year)

        # Save analysis to JSON
        filename = save_analysis_to_json(analysis, usage_stats, filename=f"{args.symbol}_analysis_Q{args.quarter}_{args.year}.json")
        if filename:
            print(f"\nAnalysis saved to {filename}")

            # Optionally, load and display the saved analysis
            load_and_display_analysis(filename)
        else:
            print("Failed to save analysis to JSON file")

    except OpenAIQuotaError as e:
        print(f"OpenAI Quota Error: {e}")
        print("Please check your OpenAI billing dashboard and ensure you have sufficient credits.")
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure all required environment variables are set in your .env file.")
    except APIError as e:
        print(f"API Error: {e}")
        print("Please check your API keys and try again.")
    except Exception as e:
        print(f"Unexpected error: {e}")

# ---------------------------
# Flask Routes
# ---------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'MSFT').upper()
        quarter = request.form.get('quarter')
        year = request.form.get('year')

        # Input validation
        if not quarter.isdigit() or not (1 <= int(quarter) <= 4):
            flash('Quarter must be an integer between 1 and 4.', 'danger')
            return redirect(url_for('index'))
        if not year.isdigit() or not (2000 <= int(year) <= datetime.now().year):
            flash(f'Year must be an integer between 2000 and {datetime.now().year}.', 'danger')
            return redirect(url_for('index'))

        quarter = int(quarter)
        year = int(year)

        try:
            analyzer = TranscriptAnalyzer(max_retries=3, retry_delay=5)
            analysis, usage_stats = analyzer.analyze_transcript(symbol=symbol, quarter=quarter, year=year)

            # Save analysis to JSON
            filename = save_analysis_to_json(analysis, usage_stats, filename=f"{symbol}_analysis_Q{quarter}_{year}.json")
            if filename:
                flash(f'Analysis successfully saved to {filename}', 'success')
                return redirect(url_for('results', filename=filename))
            else:
                flash('Failed to save analysis to JSON file.', 'danger')
                return redirect(url_for('index'))

        except OpenAIQuotaError as e:
            flash(f"OpenAI Quota Error: {e}", 'danger')
            flash("Please check your OpenAI billing dashboard and ensure you have sufficient credits.", 'info')
            return redirect(url_for('index'))
        except ConfigurationError as e:
            flash(f"Configuration Error: {e}", 'danger')
            flash("Please ensure all required environment variables are set in your .env file.", 'info')
            return redirect(url_for('index'))
        except APIError as e:
            flash(f"API Error: {e}", 'danger')
            flash("Please check your API keys and try again.", 'info')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Unexpected error: {e}", 'danger')
            return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/results/<filename>')
def results(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return render_template('results.html', data=data, datetime=datetime)
    except Exception as e:
        flash(f"Error loading analysis: {e}", 'danger')
        return redirect(url_for('index'))


# ---------------------------
# Main Entry Point
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Earnings Call Transcript Analyzer with CLI and Flask Frontend")
    parser.add_argument('--symbol', type=str, help='Stock symbol of the company')
    parser.add_argument('--quarter', type=int, help='Quarter number (1-4)')
    parser.add_argument('--year', type=int, help='Fiscal year')
    parser.add_argument('--flask', action='store_true', help='Run the Flask web frontend')

    args = parser.parse_args()

    if args.flask:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    elif args.symbol and args.quarter and args.year:
        # Run CLI mode
        try:
            analyzer = TranscriptAnalyzer(max_retries=3, retry_delay=5)
            analysis, usage_stats = analyzer.analyze_transcript(symbol=args.symbol, quarter=args.quarter, year=args.year)

            # Save analysis to JSON
            filename = save_analysis_to_json(analysis, usage_stats, filename=f"{args.symbol}_analysis_Q{args.quarter}_{args.year}.json")
            if filename:
                print(f"\nAnalysis saved to {filename}")

                # Optionally, load and display the saved analysis
                load_and_display_analysis(filename)
            else:
                print("Failed to save analysis to JSON file")

        except OpenAIQuotaError as e:
            print(f"OpenAI Quota Error: {e}")
            print("Please check your OpenAI billing dashboard and ensure you have sufficient credits.")
        except ConfigurationError as e:
            print(f"Configuration Error: {e}")
            print("Please ensure all required environment variables are set in your .env file.")
        except APIError as e:
            print(f"API Error: {e}")
            print("Please check your API keys and try again.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    else:
        print("Invalid arguments. Use --help for more information.")

if __name__ == "__main__":
    main()
