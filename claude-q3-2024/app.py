from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from anthropic import Anthropic
import time
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import backoff
import os

app = Flask(__name__)

def get_recent_transcripts(base_dir: str, cutoff_date: str, max_transcripts: int = 5) -> list:
    """Get most recent transcript files after cutoff date"""
    base_path = Path(base_dir)
    transcript_files = []
    
    cutoff = datetime.strptime(cutoff_date, '%Y-%m-%d')
    
    for company_dir in base_path.iterdir():
        if not company_dir.is_dir():
            continue
            
        for file_path in company_dir.glob('*.json'):
            if 'profile' in file_path.name:
                continue
                
            try:
                # Adjusted to correctly extract date
                date_part = file_path.name.split('_')[3]
                date_str = date_part.split()[0]  # Assuming format like '2024-10-24 20:59:01.json'
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if file_date >= cutoff:
                    transcript_files.append((file_path, file_date))
            except Exception as e:
                print(f"Error parsing date from {file_path}: {e}")
                continue
    
    # Sort by date (newest first) and take most recent n
    transcript_files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in transcript_files[:max_transcripts]]

class TranscriptAnalyzer:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    def analyze_transcript(self, content: str, symbol: str) -> dict:
        """Extract earnings call information using Claude"""
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this earnings call transcript for {symbol} and extract detailed metrics and insights into a JSON object.

You must respond with ONLY a valid JSON object containing these exact fields:

{{
    "quarter_info": {{
        "fiscal_quarter": string,
        "fiscal_year": string,
        "call_date": string,
        "symbol": string  # Added symbol here
    }},
    "financial_metrics": {{
        "revenue": {{"value": number or null, "quote": string, "yoy_change": number or null}},
        "revenue_growth": {{"value": number or null, "quote": string}},
        "eps": {{
            "gaap": {{"value": number or null, "quote": string}},
            "non_gaap": {{"value": number or null, "quote": string}},
            "yoy_change": number or null
        }},
        "gross_margin": {{"value": number or null, "quote": string}},
        "operating_margin": {{"value": number or null, "quote": string}},
        "cash_flow": {{
            "operating": {{"value": number or null, "quote": string}},
            "free": {{"value": number or null, "quote": string}}
        }},
        "cash_position": {{"value": number or null, "quote": string}}
    }},
    "segment_performance": [
        {{
            "name": string,
            "revenue": number or null,
            "growth": number or null,
            "highlights": string,
            "quote": string
        }}
    ],
    "geographic_performance": [
        {{
            "region": string,
            "revenue": number or null,
            "growth": number or null,
            "quote": string
        }}
    ],
    "guidance": {{
        "next_quarter": {{
            "revenue": {{"low": number or null, "high": number or null}},
            "eps": {{"low": number or null, "high": number or null}}
        }},
        "full_year": {{
            "revenue": {{"low": number or null, "high": number or null}},
            "eps": {{"low": number or null, "high": number or null}}
        }},
        "quote": string
    }},
    "strategic_initiatives": [
        {{
            "initiative": string,
            "description": string,
            "status": string,
            "quote": string
        }}
    ],
    "market_position": {{
        "competitors": [
            {{"name": string, "context": string, "quote": string}}
        ],
        "market_share": {{"value": number or null, "quote": string}},
        "industry_trends": [string]
    }},
    "operational_highlights": {{
        "key_metrics": [
            {{"metric": string, "value": string, "quote": string}}
        ],
        "product_launches": [
            {{"product": string, "details": string, "quote": string}}
        ],
        "partnerships": [
            {{"partner": string, "details": string, "quote": string}}
        ]
    }},
    "risk_factors": {{
        "immediate": [
            {{"risk": string, "impact": string, "quote": string}}
        ],
        "long_term": [
            {{"risk": string, "impact": string, "quote": string}}
        ]
    }},
    "management_changes": [
        {{
            "position": string,
            "change": string,
            "quote": string
        }}
    ],
    "key_quotes": {{
        "outlook": [string],
        "strategy": [string],
        "challenges": [string]
    }}
}}

Return numbers formatted as:
- Revenue in billions USD (e.g., 12.34 for $12.34B)
- Growth rates and margins as percentages (e.g., 12.5 for 12.5%)
- EPS in dollars (e.g., 1.23 for $1.23)
- Cash flows in millions USD

Use null for any values not found and explain in the quote field.
Extract the most relevant quotes that support each data point.
Maintain consistency in decimal places (2 for financial metrics).

Transcript:
{content}"""
                }]
            )
            
            # Extract text content from response
            if hasattr(response.content[0], 'text'):
                content = response.content[0].text
            else:
                content = response.content[0]
                
            # Try to parse JSON, removing any extra text
            try:
                # Find the first { and last }
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    analysis = json.loads(json_str)
                    
                    # Ensure 'symbol' is included in 'quarter_info'
                    analysis['quarter_info']['symbol'] = symbol
                    
                    return analysis
                else:
                    raise ValueError("No JSON object found in response")
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error for {symbol}: {str(e)}")
                print(f"Raw content: {content}")
                raise
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            return {
                "quarter_info": {
                    "fiscal_quarter": None,
                    "fiscal_year": None,
                    "call_date": None,
                    "symbol": symbol
                },
                "financial_metrics": {
                    "revenue": {"value": None, "quote": f"Error: {str(e)}", "yoy_change": None},
                    "revenue_growth": {"value": None, "quote": f"Error: {str(e)}"},
                    "eps": {
                        "gaap": {"value": None, "quote": f"Error: {str(e)}"},
                        "non_gaap": {"value": None, "quote": f"Error: {str(e)}"},
                        "yoy_change": None
                    },
                    "gross_margin": {"value": None, "quote": f"Error: {str(e)}"},
                    "operating_margin": {"value": None, "quote": f"Error: {str(e)}"},
                    "cash_flow": {
                        "operating": {"value": None, "quote": f"Error: {str(e)}"},
                        "free": {"value": None, "quote": f"Error: {str(e)}"}
                    },
                    "cash_position": {"value": None, "quote": f"Error: {str(e)}"}
                },
                "segment_performance": [],
                "geographic_performance": [],
                "guidance": {
                    "next_quarter": {"revenue": {"low": None, "high": None}, "eps": {"low": None, "high": None}},
                    "full_year": {"revenue": {"low": None, "high": None}, "eps": {"low": None, "high": None}},
                    "quote": f"Error: {str(e)}"
                },
                "strategic_initiatives": [],
                "market_position": {"competitors": [], "market_share": {"value": None, "quote": ""}, "industry_trends": []},
                "operational_highlights": {"key_metrics": [], "product_launches": [], "partnerships": []},
                "risk_factors": {"immediate": [], "long_term": []},
                "management_changes": [],
                "key_quotes": {"outlook": [], "strategy": [], "challenges": []}
            }
        
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    api_key = request.form.get('api_key')
    base_dir = request.form.get('base_dir', 'tech_transcripts_historical')
    
    if not api_key:
        return jsonify({'error': 'Missing API key'}), 400
        
    if not os.path.exists(base_dir):
        return jsonify({'error': f'Directory not found: {base_dir}'}), 400

    def generate_results():
        analyzer = TranscriptAnalyzer(api_key)
        results = []
        
        try:
            # Get recent transcripts
            recent_files = get_recent_transcripts(base_dir, '2024-10-05', 5)
            yield json.dumps({"status": "info", "message": f"Found {len(recent_files)} recent transcripts"}) + '\n'
            
            for file_path in recent_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    symbol = file_path.parent.name
                    date_str = file_path.name.split('_')[3].split()[0]
                    
                    yield json.dumps({
                        "status": "processing",
                        "message": f"Processing {symbol} transcript from {date_str}..."
                    }) + '\n'
                    
                    content = data.get('content', '')
                    analysis = analyzer.analyze_transcript(content, symbol)
                    
                    # Append symbol and date inside quarter_info
                    # (Already done in analyze_transcript)
                    
                    results.append(analysis)
                    
                    yield json.dumps({
                        "status": "complete",
                        "data": analysis
                    }) + '\n'
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    yield json.dumps({
                        "status": "error",
                        "message": f"Error processing {file_path}: {str(e)}"
                    }) + '\n'
                    continue
            
            # Save results
            if results:
                output_dir = Path('output')
                output_dir.mkdir(exist_ok=True)
                
                with open(output_dir / 'full_analysis.json', 'w') as f:
                    json.dump(results, f, indent=2)
                    
                # Create DataFrame for CSV export
                # Extract relevant fields
                csv_data = []
                for r in results:
                    try:
                        quarter_info = r.get('quarter_info', {})
                        financial_metrics = r.get('financial_metrics', {})
                        row = {
                            "Symbol": quarter_info.get('symbol', 'N/A'),
                            "Quarter": quarter_info.get('fiscal_quarter', 'N/A'),
                            "Fiscal Year": quarter_info.get('fiscal_year', 'N/A'),
                            "Revenue (B USD)": financial_metrics.get('revenue', {}).get('value'),
                            "Revenue Growth (%)": financial_metrics.get('revenue_growth', {}).get('value'),
                            "EPS (Non-GAAP)": financial_metrics.get('eps', {}).get('non_gaap', {}).get('value'),
                            "Operating Margin (%)": financial_metrics.get('operating_margin', {}).get('value'),
                        }
                        csv_data.append(row)
                    except Exception as e:
                        print(f"Error preparing CSV row for {r.get('quarter_info', {}).get('symbol', 'N/A')}: {e}")
                        continue
                
                df = pd.DataFrame(csv_data)
                if not df.empty:
                    df.to_csv(output_dir / 'transcript_analysis.csv', index=False)
            
            yield json.dumps({
                "status": "finished",
                "message": "Analysis complete",
                "total": len(results)
            }) + '\n'
            
        except Exception as e:
            yield json.dumps({
                "status": "error",
                "message": f"Fatal error: {str(e)}"
            }) + '\n'

    return Response(
        stream_with_context(generate_results()),
        mimetype='text/event-stream'
    )

if __name__ == "__main__":
    app.run(debug=True)
