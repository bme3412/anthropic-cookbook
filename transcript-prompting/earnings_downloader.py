# earnings_downloader.py

from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from typing import Dict, Optional
import json

class EarningsDownloader:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('FMP_API_KEY')
        self.base_url = "https://financialmodelingprep.com/api/v3/earning_call_transcript/"

        if not self.api_key:
            raise ValueError("API key not found. Please set FMP_API_KEY in your .env file.")

    def get_transcript(self, symbol: str = 'AAPL', quarter: Optional[str] = None) -> Dict:
        """
        Download earnings call transcript for a given symbol and quarter.
        """
        endpoint = f"{self.base_url}{symbol}"
        params = {
            'apikey': self.api_key,
            'quarter': quarter if quarter else None
        }

        try:
            response = requests.get(endpoint, params={k: v for k, v in params.items() if v is not None})
            response.raise_for_status()
            data = response.json()
            if not data:
                return {'error': 'No transcript data available.'}
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching transcript: {str(e)}")
            return {'error': 'Error fetching transcript data.'}

    def save_transcript(self, transcript: Dict, symbol: str = 'AAPL'):
        """Save raw transcript to JSON file."""
        current_date = datetime.now().strftime('%Y%m%d')  # e.g., '20241023'
        filename = f"{symbol.lower()}_transcript_{current_date}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
