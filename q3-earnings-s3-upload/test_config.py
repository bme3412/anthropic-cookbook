# test_config.py

from utils.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET

print(f"AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY_ID}")
print(f"AWS_SECRET_ACCESS_KEY: {'*' * len(AWS_SECRET_ACCESS_KEY)}")
print(f"AWS_S3_BUCKET: {AWS_S3_BUCKET}")
