from dotenv import load_dotenv
import os
import requests
import json
from datetime import datetime
from typing import Dict, Optional

class EarningsDownloader:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('FMP_API_KEY')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
    def get_transcript(self, symbol: str = 'AAPL', quarter: Optional[str] = None) -> Dict:
        """
        Download earnings call transcript
        """
        endpoint = f"{self.base_url}/earning_call_transcript/{symbol}"
        
        params = {
            'apikey': self.api_key,
            'quarter': quarter if quarter else None
        }
        
        try:
            response = requests.get(endpoint, params={k: v for k, v in params.items() if v is not None})
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching transcript: {str(e)}")
            return None
    
    def save_transcript(self, transcript: Dict, symbol: str = 'AAPL'):
        """Save raw transcript to JSON file"""
        filename = f"{symbol.lower()}_transcript_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    downloader = EarningsDownloader()
    transcript = downloader.get_transcript('AAPL')
    
    if transcript:
        # Save raw transcript
        downloader.save_transcript(transcript)
        
        # Print basic info
        print(f"\nDownloaded transcript for AAPL")
        if isinstance(transcript, list) and len(transcript) > 0:
            print(f"Date: {transcript[0].get('date', 'N/A')}")
            print(f"Quarter: {transcript[0].get('quarter', 'N/A')}")
            print(f"Year: {transcript[0].get('year', 'N/A')}")