# data_ingestion/fetch_transcripts.py

import requests
from utils.config import FMP_API_KEY
from utils.logger import logger
from datetime import datetime

API_BASE_URL = "https://financialmodelingprep.com/api/v3"

def is_within_quarter(date_str: str, year: int, quarter: int) -> bool:
    """
    Determines if the given date string falls within the specified year and quarter.
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        if date_obj.year != year:
            return False
        if quarter == 1 and date_obj.month in [1, 2, 3]:
            return True
        elif quarter == 2 and date_obj.month in [4, 5, 6]:
            return True
        elif quarter == 3 and date_obj.month in [7, 8, 9]:
            return True
        elif quarter == 4 and date_obj.month in [10, 11, 12]:
            return True
    except ValueError:
        logger.error(f"Invalid date format: {date_str}")
    return False

def fetch_all_transcripts(ticker: str, year: int, quarter: int):
    """
    Fetches all earnings call transcripts for a given ticker, year, and quarter.
    """
    endpoint = f"{API_BASE_URL}/earning_call_transcript/{ticker}"
    params = {
        "year": year,
        "quarter": quarter,
        "apikey": FMP_API_KEY
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        transcripts = response.json()
        logger.debug(f"API response for {ticker} Q{quarter} {year}: {transcripts}")

        # Check if the response contains an error message
        if isinstance(transcripts, dict) and "Error Message" in transcripts:
            logger.error(f"API Error for {ticker} Q{quarter} {year}: {transcripts['Error Message']}")
            return []

        if not isinstance(transcripts, list):
            logger.error(f"Unexpected API response format: {transcripts}")
            return []

        # Filter transcripts by year and quarter
        filtered_transcripts = [
            t for t in transcripts
            if t.get("date") and is_within_quarter(t["date"], year, quarter)
        ]
        logger.info(f"Fetched {len(filtered_transcripts)} transcripts for {ticker} Q{quarter} {year}")
        return filtered_transcripts
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except ValueError as ve:
        logger.error(f"JSON decoding failed: {ve}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
    return []
