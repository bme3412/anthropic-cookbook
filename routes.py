# routes.py
from flask import Blueprint, request, jsonify
import os
from rag import LightTickerRAG
import logging

api_bp = Blueprint('api', __name__)

# Initialize RAG system
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY environment variable not set")

rag = LightTickerRAG(FMP_API_KEY)

@api_bp.route('/process_ticker', methods=['POST'])
def process_ticker():
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    if not ticker:
        return jsonify({"success": False, "message": "No ticker provided"}), 400
    
    status = rag.process_ticker(ticker)
    if status["success"]:
        return jsonify(status), 200
    else:
        return jsonify(status), 400

@api_bp.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '')
    ticker = data.get('ticker', '').upper()
    
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    results = rag.query_ticker(ticker, query_text, k=3)
    return jsonify(results), 200
