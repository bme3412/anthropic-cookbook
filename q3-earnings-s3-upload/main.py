# main.py

from data_ingestion.fetch_transcripts import fetch_all_transcripts
from data_ingestion.parse_transcripts import parse_raw_transcript
from storage.s3_storage import upload_transcript
from storage.db_storage import insert_company, insert_transcript, insert_transcript_section
from utils.logger import logger
from hashlib import md5

def main():
    """
    Main function to fetch, parse, and store earnings call transcripts.
    """
    # Define your target tickers and company names
    tickers = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        # Add more tickers and company names as needed
    }

    # Define years and quarters to fetch
    years = [2020, 2021, 2022, 2023, 2024]  # Adjust based on availability
    quarters = [1, 2, 3, 4]

    for ticker, company_name in tickers.items():
        logger.info(f"Processing transcripts for {company_name} ({ticker})")
        company_id = insert_company(company_name, ticker)
        if not company_id:
            logger.error(f"Failed to insert or retrieve company ID for {ticker}")
            continue

        for year in years:
            for quarter in quarters:
                logger.info(f"Fetching transcripts for {ticker} Q{quarter} {year}")
                transcripts = fetch_all_transcripts(ticker, year, quarter)

                if not transcripts:
                    logger.warning(f"No transcripts found for {ticker} Q{quarter} {year}")
                    continue

                for transcript in transcripts:
                    # Since 'id' is not present, generate a unique transcript ID
                    unique_str = f"{ticker}_{year}_Q{quarter}_{transcript.get('date')}"
                    transcript_id = md5(unique_str.encode()).hexdigest()

                    transcript_text = transcript.get("content")
                    earnings_date = transcript.get("date")

                    if transcript_text and earnings_date:
                        # Upload raw transcript to S3
                        s3_key = upload_transcript(ticker, year, quarter, transcript_id, transcript_text)
                        if not s3_key:
                            logger.error(f"Failed to upload transcript {transcript_id} to S3")
                            continue

                        # Insert transcript metadata into DB
                        db_transcript_id = insert_transcript(company_id, earnings_date, quarter, s3_key)
                        if not db_transcript_id:
                            logger.error(f"Failed to insert transcript {transcript_id} into DB")
                            continue

                        # Parse the raw transcript
                        parsed_transcript = parse_raw_transcript(
                            raw_text=transcript_text,
                            company=company_name,
                            ticker=ticker,
                            earnings_date=earnings_date
                        )

                        # Insert transcript sections into DB
                        for section in parsed_transcript.transcript:
                            insert_transcript_section(
                                transcript_id=db_transcript_id,
                                speaker_name=section.speaker.name,
                                speaker_role=section.speaker.role or "",
                                text=section.text,
                                timestamp=section.timestamp.isoformat() if section.timestamp else None
                            )

                        logger.info(f"Successfully processed transcript ID {db_transcript_id} for {ticker}")
                    else:
                        logger.warning(f"Transcript missing required fields: {transcript}")

if __name__ == "__main__":
    main()
