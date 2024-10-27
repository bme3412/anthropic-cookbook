# test_api_key.py

import requests
from utils.config import FMP_API_KEY
from utils.logger import logger

def test_api_key():
    ticker = "AAPL"
    year = 2020
    quarter = 3
    endpoint = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}"
    params = {
        "year": year,
        "quarter": quarter,
        "apikey": FMP_API_KEY
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            if data:
                logger.info(f"API Key is valid. Received data: {data}")
            else:
                logger.warning("API Key is valid but no transcripts found for the specified parameters.")
        else:
            logger.error(f"API Error Response: {data}")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")

if __name__ == "__main__":
    test_api_key()
