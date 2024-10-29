import requests
import pandas as pd
import json
from datetime import datetime

def get_earnings_press_releases(ticker, api_key):
    """
    Fetch earnings press releases from Financial Modeling Prep API
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL')
    api_key (str): Your FMP API key
    
    Returns:
    pandas.DataFrame: Processed earnings press release data
    """
    base_url = "https://financialmodelingprep.com/api/v3"
    endpoint = f"/earning_call_transcript/{ticker}"
    
    # Construct full URL with API key
    url = f"{base_url}{endpoint}?apikey={api_key}"
    
    try:
        # Make API request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        
        # Convert to DataFrame and process
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            print(f"No earnings press releases found for {ticker}")
            return None
            
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date in descending order
        df = df.sort_values('date', ascending=False)
        
        # Format the date back to string for JSON serialization
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_json(df, filename):
    """
    Save DataFrame to JSON with proper formatting
    """
    # Convert DataFrame to list of dictionaries
    json_data = df.to_dict(orient='records')
    
    # Save with nice formatting
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "#"
    ticker = "AAPL"
    
    earnings_data = get_earnings_press_releases(ticker, API_KEY)
    
    if earnings_data is not None:
        print(f"\nFound {len(earnings_data)} earnings press releases for {ticker}")
        print("\nMost recent earnings calls:")
        print(earnings_data[['date', 'quarter', 'year']].head())
        
        # Save to JSON
        output_file = f"{ticker}_earnings_releases.json"
        save_to_json(earnings_data, output_file)
        print(f"\nData saved to {output_file}")