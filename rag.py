# rag.py
import os
import time
import logging
import requests
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import scipy.sparse as sp

class LightTickerRAG:
    def __init__(self, fmp_api_key: str):
        self.fmp_api_key = fmp_api_key
        self.graphs = {}
        self.vectorizers = {}
        self.vectors = {}
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.logger = logging.getLogger(__name__)
        
    def fetch_earnings_calls(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Fetch earnings call transcripts for a specific ticker"""
        endpoint = f"{self.base_url}/earning-call-transcript/{ticker}"
        params = {
            "apikey": self.fmp_api_key,
            "limit": limit
        }
        
        try:
            time.sleep(0.25)  # Rate limiting
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            self.logger.info(f"Fetched {len(response.json())} transcripts for {ticker}")
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return []

    def process_ticker(self, ticker: str) -> Dict:
        """Process earnings calls for a ticker and return status"""
        status = {
            "success": False,
            "message": "",
            "data": {}
        }
        
        try:
            # Clear existing data for ticker
            self.graphs[ticker] = nx.Graph()
            self.vectors[ticker] = []
            self.vectorizers[ticker] = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            
            # Fetch transcripts
            transcripts = self.fetch_earnings_calls(ticker)
            
            if not transcripts:
                status["message"] = f"No transcripts found for {ticker}"
                return status
            
            # Process all transcripts
            all_chunks = []
            chunk_metadata = []
            
            for transcript in transcripts:
                content = transcript.get('content', '')
                if not content:
                    continue
                    
                chunks = self._chunk_document(content)
                all_chunks.extend(chunks)
                
                metadata = {
                    'date': transcript.get('date'),
                    'quarter': transcript.get('quarter'),
                    'year': transcript.get('year'),
                    'ticker': ticker
                }
                chunk_metadata.extend([metadata] * len(chunks))
            
            if not all_chunks:
                status["message"] = f"No valid content found in transcripts for {ticker}"
                return status
                
            # Vectorize chunks
            vectors = self.vectorizers[ticker].fit_transform(all_chunks)
            self.vectors[ticker] = vectors  # Store the entire TF-IDF matrix
            
            # Create graph nodes
            for i, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
                node_id = f"chunk_{i}"
                self.graphs[ticker].add_node(
                    node_id,
                    text=chunk,
                    metadata=metadata
                )
            
            # Optimize graph connections using Nearest Neighbors
            self._connect_related_nodes(ticker, vectors)
            
            status["success"] = True
            status["message"] = f"Successfully processed {len(all_chunks)} chunks for {ticker}"
            status["data"] = {
                "num_transcripts": len(transcripts),
                "num_chunks": len(all_chunks),
                "dates": sorted(set(m['date'] for m in chunk_metadata if m['date']))
            }
            
            self.logger.info(status["message"])
            
        except Exception as e:
            status["message"] = f"Error processing {ticker}: {str(e)}"
            self.logger.error(status["message"])
            
        return status

    def _chunk_document(self, text: str, chunk_size: int = 200) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _connect_related_nodes(self, ticker: str, vectors: sp.csr_matrix, n_neighbors: int = 5, similarity_threshold: float = 0.3):
        """Connect each node to its top N similar nodes based on cosine similarity"""
        from sklearn.neighbors import NearestNeighbors
        
        try:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine', algorithm='brute')
            nn.fit(vectors)
            distances, indices = nn.kneighbors(vectors)
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                node_id = f"chunk_{i}"
                for d, j in zip(dist[1:], idx[1:]):  # Skip the first one (itself)
                    similarity = 1 - d
                    if similarity > similarity_threshold:
                        neighbor_id = f"chunk_{j}"
                        self.graphs[ticker].add_edge(node_id, neighbor_id, weight=float(similarity))
            self.logger.info(f"Connected related nodes for {ticker}")
        except Exception as e:
            self.logger.error(f"Error connecting nodes for {ticker}: {e}")

    def query_ticker(self, ticker: str, query: str, k: int = 3) -> List[Dict]:
        if ticker not in self.graphs:
            return []
            
        try:
            query_vector = self.vectorizers[ticker].transform([query])
            similarities = cosine_similarity(query_vector, self.vectors[ticker]).flatten()
            
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                node_id = f"chunk_{idx}"
                node_data = self.graphs[ticker].nodes[node_id]
                
                context = []
                for neighbor in self.graphs[ticker].neighbors(node_id):
                    edge_data = self.graphs[ticker].get_edge_data(node_id, neighbor)
                    neighbor_data = self.graphs[ticker].nodes[neighbor]
                    context.append({
                        'text': neighbor_data['text'],
                        'similarity': edge_data.get('weight', 0.0)
                    })
                    
                results.append({
                    'text': node_data['text'],
                    'score': float(similarities[idx]),
                    'metadata': node_data['metadata'],
                    'context': context
                })
                
            return results
        except Exception as e:
            logging.getLogger(__name__).error(f"Error querying {ticker}: {e}")
            return []
