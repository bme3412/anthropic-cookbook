import requests
import json
import os
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse
from firecrawl import FirecrawlApp

class PressReleaseFetcher:
    def __init__(self, api_key: str, firecrawl_key: str):
        self.api_key = api_key
        self.firecrawl = FirecrawlApp(api_key=firecrawl_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_full_content_from_url(self, url: str) -> Optional[str]:
        """Get full content using Firecrawl."""
        try:
            if not url:
                return None

            print(f"Fetching content from: {url}")
            result = self.firecrawl.scrape_url(url, params={'formats': ['markdown']})
            
            # Get markdown content from Firecrawl result
            if result and 'markdown' in result:
                return result['markdown'].strip()
            
            return None
            
        except Exception as e:
            print(f"Error fetching from URL: {str(e)}")
            return None

    def fetch_releases(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Fetch press releases and get full content using Firecrawl."""
        try:
            # Get initial releases from FMP
            url = f"https://financialmodelingprep.com/api/v3/press-releases/{symbol}"
            params = {
                "apikey": self.api_key,
            }
            
            print(f"Fetching releases for {symbol}...")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            releases = response.json()
            
            if not releases:
                print(f"No releases found for {symbol}")
                return []
            
            # Limit the number of releases to process
            releases = releases[:limit]
            print(f"Processing {len(releases)} most recent releases...")
            
            processed_releases = []
            for idx, release in enumerate(releases, 1):
                print(f"\nProcessing release {idx}/{limit}: {release.get('title', '')[:50]}...")
                
                # Get initial content and URL
                content = release.get('text', '')
                url = release.get('url', '').strip()
                
                # Try to get full content if URL is available
                if url:
                    full_content = self.get_full_content_from_url(url)
                    if full_content and len(full_content) > len(content):
                        content = full_content
                        print(f"Retrieved full content from {urlparse(url).netloc}")
                
                processed_release = {
                    'symbol': symbol,
                    'date': release.get('date', ''),
                    'title': release.get('title', '').strip(),
                    'content': content,
                    'url': url,
                    'content_length': len(content)
                }
                processed_releases.append(processed_release)
                time.sleep(1)  # Rate limiting
                
            return processed_releases
            
        except Exception as e:
            print(f"Error fetching releases: {str(e)}")
            return []

def save_releases(releases: List[Dict], symbol: str, output_dir: str = "press_releases"):
    """Save releases in both JSON and text formats."""
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
        date_str = release['date'].replace(':', '-').replace(' ', '_')
        file_path = os.path.join(text_dir, f"{date_str}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Symbol: {release['symbol']}\n")
            f.write(f"Date: {release['date']}\n")
            f.write(f"Title: {release['title']}\n")
            if release['url']:
                f.write(f"URL: {release['url']}\n")
            f.write(f"Content Length: {release['content_length']} characters\n")
            f.write("\n" + "="*80 + "\n\n")
            f.write(release['content'])
    
    print(f"\nSaved {len(releases)} releases for {symbol}:")
    print(f"- JSON file: {json_path}")
    print(f"- Text files: {text_dir}")

def main():
    FMP_API_KEY = '#'
    FIRECRAWL_KEY = '#'
    SYMBOLS = ['AAPL']
    OUTPUT_DIR = "press_releases"
    LIMIT = 5  # Set limit for number of releases
    
    fetcher = PressReleaseFetcher(FMP_API_KEY, FIRECRAWL_KEY)
    
    for symbol in SYMBOLS:
        releases = fetcher.fetch_releases(symbol, limit=LIMIT)
        if releases:
            save_releases(releases, symbol, OUTPUT_DIR)
            
            # Print content length statistics
            lengths = [r['content_length'] for r in releases]
            print(f"\nContent length statistics for {symbol}:")
            print(f"Min length: {min(lengths)}")
            print(f"Max length: {max(lengths)}")
            print(f"Average length: {sum(lengths)/len(lengths):.2f}")
        
        time.sleep(1)

if __name__ == "__main__":
    main()