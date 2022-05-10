"""
A simple test suite reports some basic information about Crunchbase data

Usage examples (running from the terminal):
1) To check the default dataset
python innovation_sweet_spots/tests/data_tests/test_crunchbase.py
2) To check a different data snapshot in a differently named folder (NB: should be still in inputs/data)
python innovation_sweet_spots/tests/data_tests/test_crunchbase.py --data-folder-name cb_2021
"""

from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots import logging, PROJECT_DIR
import typer


def define_data_path(folder_name: str = None):
    """
    Defines the path to Crunchbase data, given the folder name.
    NB: The folder should be located in 'inputs/data'.
    """
    if folder_name is None:
        return None
    else:
        return PROJECT_DIR / f"inputs/data/{folder_name}"


def log_total_number(CB: CrunchbaseWrangler):
    """Total number of organisations"""
    n = len(CB.cb_organisations)
    logging.info(f"There are {n} organisations in the table 'crunchbase_organisations'")


def log_uk_number(CB: CrunchbaseWrangler):
    """Organisations in the UK"""
    n_uk = len(CB.cb_organisations.query('country == "United Kingdom"'))
    logging.info(f"There are {n_uk} UK organisations")


def log_top_countries(CB: CrunchbaseWrangler):
    """Top countries by the number of organisations"""
    top_countries = (
        CB.cb_organisations.groupby("country")
        .agg(counts=("id", "count"))
        .sort_values("counts", ascending=False)
        .head(10)
    )
    logging.info(f"Top countries by the number of organisations:\n{top_countries}")


def log_number_of_industries(CB: CrunchbaseWrangler):
    """Number of unique categories/industries and industry groups"""
    n_industries = len(CB.industries)
    n_groups = len(CB.industry_groups)
    logging.info(
        f"Organisations are categorised across {n_industries} industries, which in turn are grouped into {n_groups} broader categories"
    )


def log_number_funding_rounds(CB: CrunchbaseWrangler):
    """Number of funding rounds"""
    n_rounds = len(CB.cb_funding_rounds)
    logging.info(
        f"There are {n_rounds} funding rounds in the table 'crunchbase_funding_rounds'"
    )


def log_number_people(CB: CrunchbaseWrangler):
    """Number of funding rounds"""
    n_people = len(CB.cb_people)
    logging.info(
        f"There is data about {n_people} people in the table 'crunchbase_people'"
    )


def print_all_logs(CB: CrunchbaseWrangler):
    """Performs all checks and prints logs"""
    log_total_number(CB)
    log_uk_number(CB)
    log_top_countries(CB)
    log_number_of_industries(CB)
    log_number_funding_rounds(CB)
    log_number_people(CB)


def produce_logs(data_folder_name: str = None):
    print_all_logs(CrunchbaseWrangler(define_data_path(data_folder_name)))


if __name__ == "__main__":
    typer.run(produce_logs)
