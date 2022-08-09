"""
innovation_sweet_spots.getters.nihr

Module for easy access to downloaded NIHR data
"""
# import numpy as np 
# import pandas as pd 
# import altair as alt
import os
from innovation_sweet_spots.utils.read_from_s3 import load_csv_from_s3

def get_nihr_summary_data():
    """Load NIHR .csv file from S3"""
    return load_csv_from_s3("inputs/data/nihr/", "nihr_summary_data")

def save_nihr_to_local():
    """Save NIHR summary data locally"""
    data = get_nihr_summary_data()
    path = os.path.abspath("data/") + "/nihr_summary_data.csv"
    print(path)
    data.to_csv(path, sep=",")

if __name__ == '__main__':
    # Load NIHR data and save locally 
    save_nihr_to_local()