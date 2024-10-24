# app.py
from flask import Flask, render_template, request, redirect, url_for
from earnings_downloader import EarningsDownloader
from earnings_analyzer import EarningsAnalyzer
from functools import lru_cache
import json

app = Flask(__name__)
downloader = EarningsDownloader()
analyzer = EarningsAnalyzer()

TICKER_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

@lru_cache(maxsize=32)
def get_transcript_cached(ticker):
    transcript_data = downloader.get_transcript(ticker)
    if transcript_data and isinstance(transcript_data, list) and len(transcript_data) > 0:
        downloader.save_transcript(transcript_data, ticker)
    return transcript_data

@lru_cache(maxsize=32)
def analyze_transcript_cached(ticker, analysis_type="earnings_analysis"):
    transcript_data = get_transcript_cached(ticker)
    if transcript_data and isinstance(transcript_data, list) and len(transcript_data) > 0:
        transcript_content = transcript_data[0].get('content', '')
        print(f"\nAnalyzing transcript for {ticker}...")
        result = analyzer.analyze_transcript(transcript_content, analysis_type)
        
        # Debug print
        print(f"\nAnalysis result structure:")
        print(json.dumps(result, indent=2)[:500] + "...")  # Print first 500 chars
        
        return result
    else:
        print(f"\nNo transcript content available for {ticker}")
        return {"error": "No transcript content available"}

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_ticker = request.args.get('ticker', TICKER_SYMBOLS[0])
    view_type = request.args.get('view', 'raw')
    transcript_data = None
    analysis_result = {}

    if request.method == 'POST':
        selected_ticker = request.form.get('ticker')
        view_type = request.form.get('view_type', 'raw')
        return redirect(url_for('index', ticker=selected_ticker, view=view_type))

    transcript = get_transcript_cached(selected_ticker)

    if transcript and isinstance(transcript, list) and len(transcript) > 0:
        transcript_content = transcript[0].get('content', '')
        transcript_data = {'content': transcript_content}

        if view_type == 'extracted':
            # Get analysis with debug info
            print(f"\nRequesting analysis for {selected_ticker}...")
            analysis_result = analyze_transcript_cached(selected_ticker)
            print(f"\nAnalysis result keys: {analysis_result.keys() if isinstance(analysis_result, dict) else 'Not a dict'}")
    else:
        transcript_data = {'error': 'No transcript data available.'}

    return render_template(
        'index.html',
        tickers=TICKER_SYMBOLS,
        transcript_data=transcript_data,
        metrics=analysis_result,  # Pass the full analysis result
        selected_ticker=selected_ticker,
        view_type=view_type
    )

if __name__ == '__main__':
    app.run(debug=True)