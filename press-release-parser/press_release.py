import requests
import json
from datetime import datetime
import os
import time
from typing import List, Dict, Optional

class PressReleaseFetcher:
    """Fetches and saves press releases from Financial Modeling Prep API."""
    
    def __init__(self, api_key: str, output_dir: str = "press_releases"):
        """
        Initialize the fetcher with API key and output directory.
        
        Args:
            api_key (str): FMP API key
            output_dir (str): Directory to save press releases
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3/press-releases"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def fetch_press_releases(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """
        Fetch press releases for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            limit (int): Maximum number of press releases to fetch
            
        Returns:
            List of press releases or None if request fails
        """
        try:
            url = f"{self.base_url}/{symbol}"
            params = {
                "apikey": self.api_key,
                "limit": limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching press releases for {symbol}: {str(e)}")
            return None
        
    def save_press_releases(self, symbol: str, press_releases: List[Dict]) -> str:
        """
        Save press releases to JSON file.
        
        Args:
            symbol (str): Stock symbol
            press_releases (List[Dict]): List of press releases to save
            
        Returns:
            str: Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_press_releases_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "symbol": symbol,
                "fetch_date": datetime.now().isoformat(),
                "count": len(press_releases),
                "press_releases": press_releases
            }, f, indent=2)
            
        return filepath

def main():
    # Configuration
    API_KEY = '#'  # Replace with your actual API key
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']  # Add more symbols as needed
    RATE_LIMIT_DELAY = 1  # Delay between API calls in seconds
    
    # Initialize fetcher
    fetcher = PressReleaseFetcher(API_KEY)
    
    # Process each symbol
    for symbol in SYMBOLS:
        print(f"Fetching press releases for {symbol}...")
        
        # Fetch press releases
        press_releases = fetcher.fetch_press_releases(symbol)
        
        if press_releases:
            # Save to file
            filepath = fetcher.save_press_releases(symbol, press_releases)
            print(f"Saved {len(press_releases)} press releases to {filepath}")
        else:
            print(f"No press releases retrieved for {symbol}")
            
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

if __name__ == "__main__":
    main()