# analyzer.py
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import requests
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAIError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from models import (
    TokenUsageTracker, FinancialMetric, KPI, OperationalMetric,
    RiskFactor, StrategicInitiative, EarningsAnalysis
)
from utils import PromptGenerator

class AnalyzerException(Exception):
    """Base exception for analyzer errors"""
    pass

class TranscriptAnalyzer:
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """Initialize analyzer with environment variables and retry settings"""
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.token_tracker = TokenUsageTracker()
        self.current_company = None

        if not self.fmp_api_key or not self.openai_api_key:
            raise AnalyzerException("Missing required API keys")

        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key,
                timeout=60,
                max_retries=self.max_retries
            )
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4o-mini",
                openai_api_key=self.openai_api_key,
                timeout=60,
                max_retries=self.max_retries
            )
        except OpenAIError as e:
            raise AnalyzerException(f"Failed to initialize OpenAI components: {str(e)}")

        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_transcript(self, symbol: str, quarter: int, year: int) -> str:
        """Fetch earnings call transcript"""
        base_url = os.getenv('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')
        params = {'quarter': quarter, 'year': year, 'apikey': self.fmp_api_key}
        
        try:
            response = requests.get(
                f"{base_url}/earning_call_transcript/{symbol}",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, list) or len(data) == 0:
                raise AnalyzerException("No transcript data received")

            transcript = data[0].get("content", "")
            if not transcript:
                raise AnalyzerException("Empty transcript content")

            return transcript.replace('\n', ' ').replace('\r', ' ')
        except Exception as e:
            raise AnalyzerException(f"Failed to fetch transcript: {str(e)}")

    def create_vector_store(self, transcript: str) -> FAISS:
        """Create vector store from transcript"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_text(transcript)
            
            total_tokens = sum(
                self.token_tracker.count_tokens(chunk, 'text-embedding-3-small')
                for chunk in chunks
            )
            self.token_tracker.add_usage('text-embedding-3-small', total_tokens)
            
            return FAISS.from_texts(chunks, self.embeddings)
        except Exception as e:
            raise AnalyzerException(f"Failed to create vector store: {str(e)}")

    def safe_parse_json(self, response: str | dict) -> dict:
        """Safely parse LLM response"""
        try:
            if isinstance(response, dict):
                return response
            if isinstance(response, str):
                if response.lower() == "i don't know":
                    return self._get_default_response()
                return json.loads(response.replace('```json', '').replace('```', ''))
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return self._get_default_response()

    def _get_default_response(self) -> dict:
        """Get default response structure"""
        return {
            "summary": f"Analysis unavailable for {self.current_company}",
            "highlights": ["Data extraction failed"],
            "financial_metrics": [],
            "kpis": [],
            "operational_metrics": [],
            "risk_factors": [],
            "strategic_initiatives": []
        }

    def analyze_transcript(self, symbol: str, quarter: int, year: int) -> Tuple[EarningsAnalysis, dict]:
        """Main analysis function"""
        try:
            self.current_company = symbol
            transcript = self.get_transcript(symbol, quarter, year)
            vector_store = self.create_vector_store(transcript)

            # Extract all components
            metrics = self._extract_metrics(vector_store, "financial_metrics")
            kpis = self._extract_metrics(vector_store, "kpis")
            operational_metrics = self._extract_metrics(vector_store, "operational_metrics")
            risk_factors = self._extract_metrics(vector_store, "risk_factors")
            strategic_initiatives = self._extract_metrics(vector_store, "strategic_initiatives")
            summary, highlights = self._generate_summary(vector_store)

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

            return analysis, self.token_tracker.get_total_usage()

        except Exception as e:
            raise AnalyzerException(f"Analysis failed: {str(e)}")

    def _extract_metrics(self, vector_store: FAISS, metric_type: str) -> List:
        """Generic metric extraction method"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}
                )
            )

            prompt = getattr(PromptGenerator, f"get_{metric_type}_prompt")(self.current_company)
            
            input_tokens = self.token_tracker.count_tokens(prompt, 'gpt-4o-mini')

            response = qa_chain.invoke(prompt)
            llm_result = response.get('result', '')
            
            # Track output tokens
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            # Parse and validate response
            parsed_data = self.safe_parse_json(llm_result)
            if not isinstance(parsed_data, dict) or metric_type not in parsed_data:
                return []

            # Convert to appropriate model objects
            model_map = {
                'financial_metrics': FinancialMetric,
                'kpis': KPI,
                'operational_metrics': OperationalMetric,
                'risk_factors': RiskFactor,
                'strategic_initiatives': StrategicInitiative
            }

            return [model_map[metric_type](**item) for item in parsed_data[metric_type]]
            
        except Exception as e:
            print(f"Error extracting {metric_type}: {str(e)}")
            return []

    def _generate_summary(self, vector_store: FAISS) -> Tuple[str, List[str]]:
        """Generate summary and highlights"""
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": int(os.getenv('RETRIEVER_K', '10'))}
                )
            )

            prompt = PromptGenerator.get_summary_prompt(self.current_company)
            
            input_tokens = self.token_tracker.count_tokens(prompt, 'gpt-4o-mini')
            response = qa_chain.invoke(prompt)
            llm_result = response.get('result', '')
            output_tokens = self.token_tracker.count_tokens(str(llm_result), 'gpt-4o-mini')
            
            self.token_tracker.add_usage('gpt-4o-mini', input_tokens, output_tokens)

            summary_data = self.safe_parse_json(llm_result)
            return (
                summary_data.get('summary', f"Summary unavailable for {self.current_company}"),
                summary_data.get('highlights', ["No highlights available"])
            )

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return f"Summary unavailable for {self.current_company}", ["No highlights available"]