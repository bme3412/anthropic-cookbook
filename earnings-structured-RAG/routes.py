# routes.py
from flask import Blueprint, request, jsonify
import os
from rag import StructuredEarningsRAG
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Initialize RAG system with both required API keys
FMP_API_KEY = os.getenv("FMP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not FMP_API_KEY or not OPENAI_API_KEY:
    missing_keys = []
    if not FMP_API_KEY:
        missing_keys.append("FMP_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

try:
    rag = StructuredEarningsRAG(FMP_API_KEY, OPENAI_API_KEY)
    logger.info("Successfully initialized StructuredEarningsRAG")
except Exception as e:
    logger.error(f"Failed to initialize StructuredEarningsRAG: {str(e)}\n{traceback.format_exc()}")
    raise

@api_bp.route('/process_ticker', methods=['POST'])
def process_ticker():
    try:
        data = request.get_json()
        logger.info(f"Received process_ticker request with data: {data}")
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({"success": False, "message": "No data provided"}), 400
            
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            logger.error("No ticker provided")
            return jsonify({"success": False, "message": "No ticker provided"}), 400
        
        logger.info(f"Processing ticker: {ticker}")
        
        try:
            transcripts = rag.fetch_transcripts(ticker)
            if not transcripts:
                logger.warning(f"No transcripts found for {ticker}")
                return jsonify({
                    "success": False,
                    "message": f"No transcripts found for {ticker}"
                }), 404
                
            logger.info(f"Found {len(transcripts)} transcripts for {ticker}")
            return jsonify({
                "success": True,
                "message": f"Successfully fetched {len(transcripts)} transcripts for {ticker}"
            }), 200
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "success": False,
                "message": f"Error processing ticker: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in process_ticker: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@api_bp.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        logger.info(f"Received query request with data: {data}")
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
            
        query_text = data.get('query', '')
        ticker = data.get('ticker', '').upper()
        start_date = data.get('startDate')
        end_date = data.get('endDate')
        
        if not ticker:
            logger.error("No ticker provided")
            return jsonify({"error": "No ticker provided"}), 400
        if not query_text:
            logger.error("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"Querying {ticker} with: {query_text}")
        
        try:
            if query_text.lower().startswith(("bull case", "bear case", "investment thesis")):
                results = rag.query_investment_thesis(ticker, query_text)
            else:
                results = rag.query_earnings(ticker, query_text, k=5)
            
            if not results:
                logger.warning(f"No results found for {ticker} with query: {query_text}")
                return jsonify([]), 200
            
            # Filter by date if provided
            if start_date and end_date:
                # Ensure dates are in YYYY-MM-DD format for comparison
                # If dates are in different formats, adjust accordingly
                if isinstance(results, dict):  # For investment thesis
                    for case in results:
                        filtered = [
                            r for r in results[case] 
                            if start_date <= r['metadata']['date'] <= end_date
                        ]
                        results[case] = filtered
                else:  # For regular queries
                    results = [
                        r for r in results 
                        if start_date <= r['date'] <= end_date
                    ]
            
            logger.info(f"Found results for {ticker}")
            return jsonify(results), 200
            
        except Exception as e:
            logger.error(f"Error querying {ticker}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "error": f"Error processing query: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

@api_bp.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        "error": "An unexpected error occurred",
        "message": str(error)
    }), 500
