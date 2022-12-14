"""
innovation_sweet_spots.getters.gtr_api

Module for easy access fetch data from GtR API

"""
from innovation_sweet_spots import logging
from typing import Iterator
import requests
from time import sleep
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import datetime


def get_project_info(project_id: str):
    """Call GtR API to get project data"""
    return requests.get(
        f"https://gtr.ukri.org/gtr/api/projects/{project_id}",
        headers={"Accept": "application/vnd.rcuk.gtr.json-v7"},
    )


def get_project_funds(project_id: str):
    """Call GtR API to get funding data for a specified project"""
    return requests.get(
        f"https://gtr.ukri.org/gtr/api/projects/{project_id}/funds",
        headers={"Accept": "application/vnd.rcuk.gtr.json-v7"},
    )


def get_fund_from_api_response(r: requests.models.Response):
    """Extract research funding amount from funds API response"""
    if r.json()["totalSize"] != 0:
        return r.json()["fund"][0]["valuePounds"]["amount"]
    else:
        return 0


def convert_timestamp(integer_time: int):
    return pd.Timestamp(datetime.datetime.fromtimestamp(integer_time / 1e3))


def get_start_end_from_api_response(r: requests.models.Response):
    """Extract start and end date from funds API response"""
    if r.json()["totalSize"] != 0:
        return convert_timestamp(r.json()["fund"][0]["start"]).strftime(
            "%Y-%m-%d"
        ), convert_timestamp(r.json()["fund"][0]["end"]).strftime("%Y-%m-%d")
    else:
        return str(pd.Timestamp(np.datetime64("NaT"))), str(
            pd.Timestamp(np.datetime64("NaT"))
        )


def get_funds_for_projects(
    list_of_project_ids: Iterator[str],
    filepath,
    rewrite: bool = False,
    verbose: bool = False,
    column_names: Iterator[str] = [
        "i",
        "project_id",
        "amount",
        "fund_start",
        "fund_end",
    ],
) -> pd.DataFrame:
    """ """
    # Check if file exists, and make calls for only the necessary projects
    if filepath.exists() and (not rewrite):
        j = len(pd.read_csv(filepath, names=column_names))
    else:
        j = 0
        # with open(filepath, 'w') as f: pass

    # If all projects are not covered
    if j != len(list_of_project_ids):
        # Project ids to check
        ids = list_of_project_ids[j:]
        with open(filepath, "a") as f:
            writer = csv.writer(f)
            for i, project_id in tqdm(enumerate(ids), total=len(ids)):
                if verbose:
                    logging.info(f"Project ID: {project_id}")
                r = get_project_funds(project_id)
                fund_amount = get_fund_from_api_response(r)
                start, end = get_start_end_from_api_response(r)
                writer.writerow([i + j, project_id, fund_amount, start, end])
                sleep(0.05)

    df = pd.read_csv(filepath, names=column_names)
    assert sorted(list_of_project_ids) == sorted(df.project_id.to_list())
    return df
