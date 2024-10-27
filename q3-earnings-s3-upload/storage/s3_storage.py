# storage/s3_storage.py

import boto3
from utils.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET
from utils.logger import logger
from botocore.exceptions import NoCredentialsError, ClientError

def get_s3_client():
    """
    Initializes and returns an S3 client.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        return s3_client
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
        return None

def upload_transcript(ticker: str, year: int, quarter: int, transcript_id: str, transcript_text: str):
    """
    Uploads the raw transcript text to S3.
    Returns the S3 key if successful, else None.
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None

    file_key = f"transcripts/{ticker}/{year}/Q{quarter}/{transcript_id}.txt"
    try:
        s3_client.put_object(Bucket=AWS_S3_BUCKET, Key=file_key, Body=transcript_text)
        logger.info(f"Uploaded transcript to s3://{AWS_S3_BUCKET}/{file_key}")
        return file_key
    except ClientError as e:
        logger.error(f"Failed to upload transcript: {e}")
    return None
