"""
innovation_sweet_spots.utils.nihr.create_nihr

Module for creating NIHR data from source and saving into the S3 bucket
"""
import numpy as np 
import pandas as pd 
import altair as alt
import sys
from innovation_sweet_spots.utils.read_from_s3 import save_csv_to_s3

COLUMNS = ['datasetid', 'recordid', 'geometry', 'record_timestamp',
       'latitude', 'postcode', 'programme_stream', 'funder', 'longitude',
       'project_status', 'project_title', 'award_holder_name', 'start_date',
       'acronym', 'funding_stream', 'orcid', 'contracted_organisation', 'geo',
       'programme_type', 'award_amount_from_dh', 'scientific_abstract',
       'programme', 'project_id', 'involvement_type', 'ukcrc_value',
       'end_date', 'plain_english_abstract', 'organisation_type',
       'award_amount_m']

def create_nihr():
    """Load NIHR data from source and wrangle to get appropriate format"""
    records = pd.read_json(
        "https://nihr.opendatasoft.com/explore/dataset/nihr-summary-view/download/?format=json&timezone=Europe/London&lang=en"
        )
    # Collect field that has main information, extract and create columns for each nested field
    keys = records['fields'].iloc[0].keys()
    for k in keys: 
        records[k] = records['fields'].apply(lambda x:x.get(k))
    # Select relevant columns
    records = records[COLUMNS]
    return records 

if __name__ == '__main__':
    data = create_nihr()
    #Save to S3 
    save_csv_to_s3("inputs/data/nihr", "nihr_summary_data", data)