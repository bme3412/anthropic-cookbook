from flask import Flask, request, render_template
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import time
import logging
from typing import List, Dict
import os
import nltk
from flask_caching import Cache
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'flask')

def initialize_nltk():
    """Initialize NLTK resources with proper error handling"""
    try:
        # Download the standard punkt tokenizer
        nltk.download('punkt')
        logger.info("Successfully downloaded NLTK punkt tokenizer")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        raise RuntimeError("Failed to initialize NLTK resources")

# Initialize NLTK resources before starting the app
initialize_nltk()

class LightTickerRAG:
    def __init__(self, fmp_api_key: str, openai_api_key: str):
        self.fmp_api_key = fmp_api_key
        self.openai_api_key = openai_api_key
        self.graphs = {}
        self.embeddings = {}
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.headers = {
            'Authorization': f'Bearer {openai_api_key}',  # Correctly formatted
            'Content-Type': 'application/json'
        }

    @cache.memoize(timeout=3600)
    def fetch_earnings_calls(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Fetch earnings call transcripts with caching"""
        endpoint = f"{self.base_url}/earning_call_transcript/{ticker}"
        params = {
            "apikey": self.fmp_api_key,
            "limit": limit
        }

        try:
            logger.info(f"Fetching transcripts for {ticker}")
            time.sleep(0.25)  # Rate limiting
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Retrieved {len(data)} transcripts for {ticker}")
            return data
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred for {ticker}: {http_err}")
            return []
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return []

    def _chunk_document(self, text: str, max_tokens: int = 200) -> List[str]:
        """Split document into chunks using simple sentence splitting"""
        try:
            # Use nltk's punkt tokenizer for sentence splitting
            sentences = nltk.sent_tokenize(text)  # Ensure 'punkt' is used
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks if chunks else [text]
        except Exception as e:
            logger.error(f"Error in document chunking: {str(e)}")
            # Fallback to simple chunking if NLTK fails
            words = text.split()
            chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
            return chunks if chunks else [text]

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API with improved error handling"""
        logger.info("Generating embeddings using OpenAI API")
        embeddings = []
        try:
            for text in texts:
                response = requests.post(
                    'https://api.openai.com/v1/embeddings',
                    headers=self.headers,
                    json={
                        'input': text,
                        'model': 'text-embedding-ada-002'
                    }
                )
                
                if response.status_code == 401:
                    logger.error("OpenAI API authentication failed. Please check your API key.")
                    return np.array([])
                
                response.raise_for_status()
                data = response.json()
                embedding = data['data'][0]['embedding']
                embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
            
            return np.array(embeddings)
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return np.array([])
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.array([])

    @cache.memoize(timeout=86400)  # Cache summaries for 1 day
    def _generate_summary(self, text: str) -> str:
        """Generate a summary for a given text using OpenAI's Chat Completion API."""
        logger.info("Generating summary using OpenAI API")
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',  # Updated endpoint
                headers=self.headers,
                json={
                    'model': 'gpt-3.5-turbo',  # Use a chat-based model
                    'messages': [
                        {"role": "system", "content": "You are a helpful assistant that summarizes texts."},
                        {"role": "user", "content": f"Provide a concise summary for the following text:\n\n{text}"}
                    ],
                    'max_tokens': 150,
                    'temperature': 0.5,
                }
            )
            
            if response.status_code == 401:
                logger.error("OpenAI API authentication failed. Please check your API key.")
                return "Summary generation failed due to authentication error."
            
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content'].strip()
            return summary
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return "Summary generation failed due to a request error."
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary generation failed due to an unexpected error."

    def process_ticker(self, ticker: str) -> Dict:
        """Process ticker data using OpenAI APIs"""
        logger.info(f"Processing ticker {ticker}")
        status = {
            "success": False,
            "message": "",
            "data": {}
        }

        try:
            if ticker not in self.graphs:
                self.graphs[ticker] = nx.Graph()
                self.embeddings[ticker] = []

                transcripts = self.fetch_earnings_calls(ticker)
                if not transcripts:
                    status["message"] = f"No transcripts found for {ticker}"
                    return status

                all_chunks = []
                chunk_metadata = []

                for transcript in transcripts:
                    content = transcript.get('content', '').strip()
                    if not content:
                        continue

                    chunks = self._chunk_document(content)
                    all_chunks.extend(chunks)
                    
                    metadata = {
                        'date': transcript.get('date', 'N/A'),
                        'quarter': transcript.get('quarter', 'N/A'),
                        'year': transcript.get('year', 'N/A'),
                        'ticker': ticker
                    }
                    chunk_metadata.extend([metadata] * len(chunks))

                if not all_chunks:
                    status["message"] = f"No valid content found for {ticker}"
                    return status

                embeddings = self._generate_embeddings(all_chunks)
                if embeddings.size == 0:
                    status["message"] = "Failed to generate embeddings"
                    return status

                self.embeddings[ticker] = embeddings

                # Create graph nodes and edges
                for i, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
                    node_id = f"chunk_{i}"
                    self.graphs[ticker].add_node(node_id, text=chunk, metadata=metadata)
                    
                    if i > 0:
                        for j in range(i):
                            prev_id = f"chunk_{j}"
                            similarity = float(cosine_similarity(
                                [embeddings[i]], [embeddings[j]]
                            )[0][0])
                            if similarity > 0.3:
                                self.graphs[ticker].add_edge(node_id, prev_id, weight=similarity)

                status["success"] = True
                status["message"] = f"Successfully processed {ticker}"
                status["data"] = {
                    "num_transcripts": len(transcripts),
                    "num_chunks": len(all_chunks)
                }
            else:
                status["success"] = True
                status["message"] = f"Data already processed for {ticker}"

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            status["message"] = f"Error processing {ticker}: {str(e)}"

        return status

    def query_ticker(self, ticker: str, query: str, k: int = 10) -> List[Dict]:
        """Query processed ticker data for relevant information based on similarity to query."""
        if ticker not in self.embeddings:
            logger.error(f"No embeddings found for ticker {ticker}")
            return []

        try:
            # Generate an embedding for the query using OpenAI API
            query_embedding = self._generate_embeddings([query])
            if query_embedding.size == 0:
                logger.error("Failed to generate embedding for the query")
                return []

            similarities = cosine_similarity(query_embedding, self.embeddings[ticker])
            sorted_indices = np.argsort(similarities[0])[::-1][:k]  # Top-k similar chunks

            results = []
            for idx in sorted_indices:
                node_id = f"chunk_{idx}"
                node_data = self.graphs[ticker].nodes[node_id]
                summary = self._generate_summary(node_data['text'])
                results.append({
                    "text": node_data['text'],
                    "summary": summary,
                    "metadata": node_data['metadata'],
                    "score": similarities[0][idx]
                })

            return results
        except Exception as e:
            logger.error(f"Error querying ticker {ticker}: {str(e)}")
            return []

# Initialize RAG system with API keys from environment variables
FMP_API_KEY = os.environ.get('FMP_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not FMP_API_KEY or not OPENAI_API_KEY:
    error_message = []
    if not FMP_API_KEY:
        error_message.append("FMP_API_KEY")
    if not OPENAI_API_KEY:
        error_message.append("OPENAI_API_KEY")
    raise ValueError(f"Missing required environment variables: {', '.join(error_message)}")

try:
    rag = LightTickerRAG(FMP_API_KEY, OPENAI_API_KEY)
    logger.info("Successfully initialized LightTickerRAG")
except Exception as e:
    logger.error(f"Failed to initialize LightTickerRAG: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def process_search():
    try:
        ticker = request.form.get('ticker', '').strip().upper()
        query = request.form.get('query', '').strip()
        start_date = request.form.get('start_date', '').strip()
        end_date = request.form.get('end_date', '').strip()

        if not ticker or not query:
            return render_template('index.html', error="Please provide both ticker and query")

        logger.info(f"Processing search for {ticker}: {query}")
        status = rag.process_ticker(ticker)

        if not status['success']:
            return render_template('index.html', error=status['message'])

        results = rag.query_ticker(ticker, query, k=10)

        if start_date and end_date:
            results = [r for r in results if start_date <= r['metadata']['date'] <= end_date]

        if not results:
            return render_template('index.html', error=f"No results found for {ticker}")

        return render_template('index.html', results=results, ticker=ticker, query=query)

    except Exception as e:
        logger.error(f"Error processing search: {str(e)}", exc_info=True)
        return render_template('index.html', error="An error occurred processing your request")

if __name__ == '__main__':
    logger.info("Starting application...")
    app.run(debug=True)
