# utils/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")
