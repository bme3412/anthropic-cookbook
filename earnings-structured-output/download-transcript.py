import requests
import json
from datetime import datetime
import os
import time

def get_fmp_transcript(ticker, api_key):
    """
    Download the most recent earnings call transcript from Financial Modeling Prep API.
    
    Args:
        ticker (str): Company ticker symbol
        api_key (str): FMP API key
        
    Returns:
        dict: Transcript data including metadata
    """
    try:
        # FMP API endpoint for transcripts
        base_url = "https://financialmodelingprep.com/api/v3"  # Changed to v3
        endpoint = f"/earning_call_transcript/{ticker.upper()}"
        
        # Construct URL with API key
        url = f"{base_url}{endpoint}?apikey={api_key}"
        
        print(f"Attempting to fetch transcript from: {url.replace(api_key, 'API_KEY_HIDDEN')}")
        
        # Make API request
        response = requests.get(url)
        
        # Print response status and content for debugging
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 401:
            print("Error: Invalid or missing API key")
            return None
        elif response.status_code == 403:
            print("Error: API key doesn't have access to this endpoint")
            return None
        elif response.status_code == 429:
            print("Error: Rate limit exceeded")
            return None
            
        response.raise_for_status()
        
        # Parse response
        transcript_data = response.json()
        
        if not transcript_data:
            print(f"No transcript found for {ticker}")
            print("API Response:", transcript_data)
            return None
            
        # Get the most recent transcript (first in the list)
        latest_transcript = transcript_data[0] if isinstance(transcript_data, list) else transcript_data
        
        # Format data for saving
        formatted_data = {
            "ticker": ticker.upper(),
            "download_date": datetime.now().strftime("%Y%m%d"),
            "transcript": {
                "date": latest_transcript.get("date"),
                "content": latest_transcript.get("content"),
                "symbol": latest_transcript.get("symbol")
            }
        }
        
        # Create filename with date
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{ticker.lower()}_transcript_{timestamp}.json"
        
        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=4, ensure_ascii=False)
            
        print(f"Transcript saved to {filename}")
        return formatted_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}")
        print("Raw response:", response.text[:200])  # Print first 200 chars of response
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def main():
    # Get API key from environment variable or user input
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        api_key = input("Enter your FMP API key: ").strip()
    
    if not api_key:
        print("Error: API key is required")
        return
        
    # Get ticker from user
    ticker = input("Enter company ticker symbol: ").strip()
    
    if not ticker:
        print("Error: Please enter a ticker symbol")
        return
    
    # Add rate limiting
    time.sleep(1)
    
    # Download transcript
    transcript = get_fmp_transcript(ticker, api_key)
    
    if transcript:
        print(f"Successfully downloaded transcript for {ticker}")
    else:
        print("Failed to download transcript")

if __name__ == "__main__":
    main()