import requests
import json
from datetime import datetime
import os
from typing import List, Dict, Optional
import time

class PressReleaseFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3/press-releases"

    def fetch_single_release(self, symbol: str, title: str, date: str) -> Optional[str]:
        """
        Fetch a single press release with full content using additional parameters.
        Some APIs require specific parameters to get full content.
        """
        try:
            params = {
                "apikey": self.api_key,
                "symbol": symbol,
                "title": title,
                "date": date
            }
            
            response = requests.get(f"{self.base_url}/{symbol}/single", params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                return data[0].get('text', '')
            return None
            
        except Exception as e:
            print(f"Error fetching single release: {str(e)}")
            return None

    def fetch_releases(self, symbol: str) -> List[Dict]:
        """Fetch all press releases with full content."""
        try:
            # First, get list of releases
            params = {
                "apikey": self.api_key,
                "limit": 100
            }
            
            print(f"Fetching releases for {symbol}...")
            response = requests.get(f"{self.base_url}/{symbol}", params=params)
            response.raise_for_status()
            releases = response.json()
            
            if not releases:
                print(f"No releases found for {symbol}")
                return []
            
            # Process each release
            processed_releases = []
            for release in releases:
                # Try to get full content if needed
                content = release.get('text', '')
                if len(content) < 1000:  # If content seems truncated
                    full_content = self.fetch_single_release(
                        symbol,
                        release.get('title', ''),
                        release.get('date', '')
                    )
                    if full_content:
                        content = full_content
                
                processed_release = {
                    'symbol': symbol,
                    'date': release.get('date', ''),
                    'title': release.get('title', '').strip(),
                    'content': content,
                    'url': release.get('url', '')
                }
                processed_releases.append(processed_release)
                
                # Rate limiting
                time.sleep(0.5)
            
            return processed_releases
            
        except Exception as e:
            print(f"Error fetching releases: {str(e)}")
            return []

def save_releases(releases: List[Dict], symbol: str, output_dir: str = "press_releases"):
    """Save releases in both JSON and individual text files."""
    # Create output directories
    json_dir = os.path.join(output_dir, "json")
    text_dir = os.path.join(output_dir, "text", symbol)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(json_dir, f"{symbol}_releases.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(releases, f, indent=2, ensure_ascii=False)
    
    # Save individual text files
    for release in releases:
        # Create filename from date
        date_str = release['date'].replace(':', '-').replace(' ', '_')
        file_path = os.path.join(text_dir, f"{date_str}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Symbol: {release['symbol']}\n")
            f.write(f"Date: {release['date']}\n")
            f.write(f"Title: {release['title']}\n")
            if release['url']:
                f.write(f"URL: {release['url']}\n")
            f.write("\n" + "="*80 + "\n\n")
            f.write(release['content'])
    
    print(f"Saved {len(releases)} releases for {symbol}:")
    print(f"- JSON file: {json_path}")
    print(f"- Text files: {text_dir}")

def main():
    # Configuration
    API_KEY = '#'  # Replace with your actual API key
    SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']
    OUTPUT_DIR = "press_releases"
    
    # Initialize fetcher
    fetcher = PressReleaseFetcher(API_KEY)
    
    # Process each symbol
    for symbol in SYMBOLS:
        # Fetch releases
        releases = fetcher.fetch_releases(symbol)
        
        if releases:
            # Save releases
            save_releases(releases, symbol, OUTPUT_DIR)
            
            # Print some statistics
            lengths = [len(r['content']) for r in releases]
            print(f"\nContent length statistics for {symbol}:")
            print(f"Min length: {min(lengths)}")
            print(f"Max length: {max(lengths)}")
            print(f"Average length: {sum(lengths)/len(lengths):.2f}")
        
        print("\n" + "="*80 + "\n")
        time.sleep(1)  # Rate limiting between symbols

if __name__ == "__main__":
    main()