"""
innovation_sweet_spots.utils.nihr.read_from_s3

Functions for saving and loading data into the S3 buckey
"""
import json
import boto3
import io
import pandas as pd
from pathlib import Path
from functools import lru_cache

BUCKET_NAME = "innovation-sweet-spots-lake" # put in .config

def save_to_s3(s3_path, filename, contents):
    """Saves the contents to the filename in {BUCKET_NAME}/{s3_path}"""
    s3 = boto3.resource("s3")
    obj = s3.Object(BUCKET_NAME, str(Path(s3_path) / filename))
    obj.put(Body=contents)


@lru_cache()
def load_from_s3(s3_path, filename):
    """Loads the file contents from the filename at {BUCKET_NAME}/{S3_PATH}"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=str(Path(s3_path) / filename))
    return obj["Body"].read()#.decode()


def save_csv_to_s3(s3_path, prefix, data):
    """Save data as csv on S3"""
    save_to_s3(s3_path, f"{prefix}.csv", data.to_csv(None, sep=","))


@lru_cache()
def load_csv_from_s3(s3_path, prefix):
    """Save data as csv from S3"""
    return pd.read_csv(io.BytesIO(load_from_s3(s3_path, f"{prefix}.csv")), encoding="ISO-8859-1")

