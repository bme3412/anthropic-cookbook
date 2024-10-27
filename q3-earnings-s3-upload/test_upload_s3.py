# test_upload_s3.py

import boto3
from utils.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET
from utils.logger import logger

def test_s3_upload():
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        test_content = "This is a test upload."
        s3_client.put_object(Bucket=AWS_S3_BUCKET, Key='test_upload.txt', Body=test_content)
        logger.info("Successfully uploaded test_upload.txt to S3.")
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")

if __name__ == "__main__":
    test_s3_upload()
