
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ACCESS_KEY = os.environ["RUSTFS_ACCESS_KEY"]
SECRET_KEY = os.environ["RUSTFS_SECRET_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
    config=Config(
        s3={"addressing_style": "path"}
    ),
)

BUCKETS = [
    "faceops-datasets",
    "faceops-mlflow",
]

print("\n===== FACEOPS STORAGE SETUP =====")

existing = {
    bucket["Name"]
    for bucket in s3.list_buckets()["Buckets"]
}

for bucket in BUCKETS:
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)
        print(f"CREATED: {bucket}")
    else:
        print(f"EXISTS:  {bucket}")

print("\nAvailable buckets:")
for bucket in s3.list_buckets()["Buckets"]:
    print(" -", bucket["Name"])

print("\nStorage setup completed.")
