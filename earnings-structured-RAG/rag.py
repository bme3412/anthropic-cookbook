# rag.py
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import os
import requests
import time

@dataclass
class Speaker:
    name: str
    role: Optional[str] = None
    company: Optional[str] = None

@dataclass
class TranscriptSegment:
    speaker: Speaker
    content: str
    segment_type: str  # 'presentation' or 'qa'
    timestamp: Optional[datetime] = None

class StructuredEarningsRAG:
    def __init__(self, fmp_api_key: str, openai_api_key: str):
        self.fmp_api_key = fmp_api_key
        self.openai_api_key = openai_api_key
        self.graphs = {}
        self.embeddings = {}
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json'
        }
        self.logger = logging.getLogger(__name__)
        
        # Verify NLTK data is available
        self._verify_nltk_data()

        # Add API validation
        self._validate_api_keys()

    def _validate_api_keys(self):
        """Validate API keys on initialization"""
        if not self.fmp_api_key or not self.openai_api_key:
            self.logger.error("Missing required API keys")
            raise ValueError("Both FMP and OpenAI API keys are required")

    def _verify_nltk_data(self):
        """Verify NLTK data is available without downloading"""
        nltk_data_dir = str(Path.home() / 'nltk_data' / 'tokenizers' / 'punkt')
        if not os.path.exists(nltk_data_dir):
            self.logger.error("NLTK punkt tokenizer not found. Please initialize NLTK before creating RAG instance.")
            raise RuntimeError("NLTK data not initialized")

    def fetch_transcripts(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Fetch earnings call transcripts for a specific ticker"""
        endpoint = f"{self.base_url}/earning_call_transcript/{ticker}"
        params = {
            "apikey": self.fmp_api_key,
            "limit": limit
        }
        
        try:
            self.logger.info(f"Fetching transcripts for {ticker}")
            time.sleep(0.25)  # Rate limiting
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Retrieved {len(data)} transcripts for {ticker}")
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching transcripts for {ticker}: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching transcripts for {ticker}: {str(e)}")
            return []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        try:
            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=self.headers,
                json={
                    'input': text,
                    'model': 'text-embedding-ada-002'
                }
            )
            
            if response.status_code == 401:
                self.logger.error("OpenAI API authentication failed")
                return np.array([])
                
            response.raise_for_status()
            data = response.json()
            embedding = data['data'][0]['embedding']
            
            time.sleep(0.1)  # Rate limiting
            return np.array(embedding)
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return np.array([])

    def _generate_summary(self, text: str) -> str:
        """Generate a concise summary using OpenAI's API"""
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=self.headers,
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {"role": "system", "content": "Summarize the key points from this earnings call segment in one sentence."},
                        {"role": "user", "content": text}
                    ],
                    'max_tokens': 150,
                    'temperature': 0.5
                }
            )
            
            if response.status_code == 401:
                self.logger.error("OpenAI API authentication failed")
                return ""
                
            response.raise_for_status()
            summary = response.json()['choices'][0]['message']['content'].strip()
            self.logger.info(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return ""

    def _analyze_concepts(self, text: str, concept_type: str) -> str:
        """Generate concept analysis using OpenAI's API"""
        try:
            prompt = f"Extract key {concept_type} points from this earnings call segment. Focus only on clear factual evidence rather than speculation.\n\nTranscript:\n{text}"
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=self.headers,
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {"role": "system", "content": "You are an expert financial analyst extracting key investment insights."},
                        {"role": "user", "content": prompt}
                    ],
                    'max_tokens': 150,
                    'temperature': 0.3
                }
            )
            
            if response.status_code == 401:
                self.logger.error("OpenAI API authentication failed")
                return ""
                
            response.raise_for_status()
            analyzed_concepts = response.json()['choices'][0]['message']['content'].strip()
            self.logger.info(f"Analyzed {concept_type} concepts: {analyzed_concepts}")
            return analyzed_concepts
        except Exception as e:
            self.logger.error(f"Error in concept analysis: {str(e)}")
            return ""

    def _parse_transcript(self, raw_text: str) -> List[TranscriptSegment]:
        """Parse raw transcript text into structured segments."""
        sections = self._split_into_sections(raw_text)
        segments = []
        
        for section in sections:
            speaker_blocks = self._extract_speaker_blocks(section['content'])
            
            for speaker, content in speaker_blocks:
                segment = TranscriptSegment(
                    speaker=self._parse_speaker(speaker),
                    content=self._clean_content(content),
                    segment_type=section['type']
                )
                segments.append(segment)
                
        return segments

    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split transcript into presentation and Q&A sections."""
        sections = []
        
        qa_patterns = [
            r"Question-and-Answer Session",
            r"Questions and Answers",
            r"Q&A Session",
            r"Operator:"
        ]
        
        qa_start = len(text)
        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.start() < qa_start:
                qa_start = match.start()
        
        presentation = text[:qa_start].strip()
        qa = text[qa_start:].strip()
        
        if presentation:
            sections.append({
                'type': 'presentation',
                'content': presentation
            })
        
        if qa:
            sections.append({
                'type': 'qa',
                'content': qa
            })
            
        return sections

    def _extract_speaker_blocks(self, text: str) -> List[tuple]:
        """Extract speaker blocks from text."""
        speaker_pattern = r'([A-Z][A-Za-z\s\.-]+)(?:\s*\(.*?\))?\s*:'
        
        blocks = []
        current_speaker = None
        current_content = []
        
        for line in text.split('\n'):
            speaker_match = re.match(speaker_pattern, line)
            
            if speaker_match:
                if current_speaker and current_content:
                    blocks.append((current_speaker, ' '.join(current_content)))
                    
                current_speaker = speaker_match.group(1)
                current_content = [line[speaker_match.end():].strip()]
            elif line.strip() and current_speaker:
                current_content.append(line.strip())
                
        if current_speaker and current_content:
            blocks.append((current_speaker, ' '.join(current_content)))
            
        return blocks

    def _parse_speaker(self, speaker_text: str) -> Speaker:
        """Parse speaker information into structured format."""
        role_patterns = {
            'CEO': r'(?:Chief Executive Officer|CEO|Chief Executive)',
            'CFO': r'(?:Chief Financial Officer|CFO)',
            'COO': r'(?:Chief Operating Officer|COO)',
            'Analyst': r'(?:Analyst|Research)',
            'Operator': r'Operator'
        }
        
        role = None
        company = None
        
        for role_name, pattern in role_patterns.items():
            if re.search(pattern, speaker_text, re.IGNORECASE):
                role = role_name
                break
                
        company_match = re.search(r'\((.*?)\)', speaker_text)
        if company_match:
            company = company_match.group(1)
            
        return Speaker(
            name=speaker_text.split('(')[0].strip(),
            role=role,
            company=company
        )

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text."""
        content = re.sub(r'\s+', ' ', content)
        content = content.replace('[', '').replace(']', '')
        return content.strip()

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate or highly similar content from results"""
        deduplicated = []
        seen_content = set()
        
        for result in results:
            content_key = ' '.join(result['text'].lower().split())[:100]
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(result)
        
        return deduplicated

    def query_earnings(self, ticker: str, query: str, k: int = 5) -> List[Dict]:
        """Query earnings call transcripts with improved context and structure."""
        transcripts = self.fetch_transcripts(ticker)
        if not transcripts:
            return []
            
        results = []
        for transcript in transcripts:
            segments = self._parse_transcript(transcript['content'])
            
            query_embedding = self._get_embedding(query)
            if query_embedding.size == 0:
                continue  # Skip if embedding failed
                
            segment_embeddings = [
                self._get_embedding(segment.content) 
                for segment in segments
            ]
            
            # Filter out empty embeddings
            valid_segments = []
            valid_embeddings = []
            for seg, emb in zip(segments, segment_embeddings):
                if emb.size != 0:
                    valid_segments.append(seg)
                    valid_embeddings.append(emb)
            
            if not valid_embeddings:
                continue  # No valid embeddings
            
            similarities = cosine_similarity(
                [query_embedding], 
                valid_embeddings
            )[0]
            
            top_indices = similarities.argsort()[-k:][::-1]
            
            for idx in top_indices:
                segment = valid_segments[idx]
                context = self._get_segment_context(valid_segments, idx)
                summary = self._generate_summary(segment.content)
                
                result = {
                    'text': segment.content,
                    'summary': summary,
                    'speaker': {
                        'name': segment.speaker.name,
                        'role': segment.speaker.role,
                        'company': segment.speaker.company
                    },
                    'segment_type': segment.segment_type,
                    'score': float(similarities[idx]),
                    'date': transcript.get('date', ''),
                    'quarter': transcript.get('quarter', ''),
                    'year': transcript.get('year', ''),
                    'context': context
                }
                
                results.append(result)
        
        results = self._deduplicate_results(results)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:k]

    def _get_segment_context(self, segments: List[TranscriptSegment], idx: int, 
                           context_window: int = 1) -> List[Dict]:
        """Get surrounding context for a segment."""
        context = []
        
        start_idx = max(0, idx - context_window)
        for i in range(start_idx, idx):
            context.append({
                'position': 'previous',
                'content': segments[i].content,
                'speaker': {
                    'name': segments[i].speaker.name,
                    'role': segments[i].speaker.role
                }
            })
            
        end_idx = min(len(segments), idx + context_window + 1)
        for i in range(idx + 1, end_idx):
            context.append({
                'position': 'next',
                'content': segments[i].content,
                'speaker': {
                    'name': segments[i].speaker.name,
                    'role': segments[i].speaker.role
                }
            })
            
        return context

    def query_investment_thesis(self, ticker: str, query: str) -> Dict[str, List[Dict]]:
        """Query earnings calls for bull/bear thesis analysis"""
        transcripts = self.fetch_transcripts(ticker)
        if not transcripts:
            self.logger.info("No transcripts found for investment thesis.")
            return {"bull_case": [], "bear_case": []}  # Updated keys
                
        results = {"bull_case": [], "bear_case": []}  # Updated keys
        
        for transcript in transcripts:
            segments = self._parse_transcript(transcript['content'])
            
            for segment in segments:
                if segment.speaker.role in ['CEO', 'CFO']:
                    bull_points = self._analyze_concepts(segment.content, "bullish")
                    bear_points = self._analyze_concepts(segment.content, "bearish")
                    
                    if bull_points:
                        self.logger.info(f"Adding bull points: {bull_points}")
                        results["bull_case"].append({
                            'points': bull_points,
                            'speaker': {
                                'name': segment.speaker.name,
                                'role': segment.speaker.role
                            },
                            'metadata': {
                                'date': transcript.get('date', ''),
                                'quarter': transcript.get('quarter', ''),
                                'year': transcript.get('year', '')
                            }
                        })
                    
                    if bear_points:
                        self.logger.info(f"Adding bear points: {bear_points}")
                        results["bear_case"].append({
                            'points': bear_points,
                            'speaker': {
                                'name': segment.speaker.name,
                                'role': segment.speaker.role
                            },
                            'metadata': {
                                'date': transcript.get('date', ''),
                                'quarter': transcript.get('quarter', ''),
                                'year': transcript.get('year', '')
                            }
                        })
        
        self.logger.info(f"Investment Thesis Results: {results}")
        return results
