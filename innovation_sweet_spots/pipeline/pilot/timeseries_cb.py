"""
Script to generate time series from Crunchbase data
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import import_config
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd

# Input files
DATA_DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/"
ORGS_TABLE = "ISS_pilot_Crunchbase_companies.csv"
DEALS_TABLE = "ISS_pilot_Crunchbase_deals.csv"
# Output files
EXPORT_DIR = DATA_DIR / "time_series/"
OUTFILE_NAME = "Time_series_Crunchbase_{}.csv"
# Parameters (tech categories to process, time series limits)
PARAMS = import_config("iss_pilot.yaml")

if __name__ == "__main__":

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    cb_orgs = pd.read_csv(DATA_DIR / ORGS_TABLE)
    cb_deals = pd.read_csv(DATA_DIR / DEALS_TABLE)

    # Technology categories to process
    categories = PARAMS["technology_categories"]

    for category in categories:
        # Select rows relevant to the category
        category_cb_orgs = cb_orgs.query(f'tech_category=="{category}"').copy()
        category_deals = cb_deals.query(f'tech_category=="{category}"').copy()

        # Generate time series
        time_series_investment = au.cb_get_all_timeseries(
            category_cb_orgs,
            category_deals,
            min_year=PARAMS["min_year"],
            max_year=PARAMS["max_year"],
        )

        # Export
        time_series_investment.to_csv(
            EXPORT_DIR / OUTFILE_NAME.format(category), index=False
        )

    logging.info(
        f"Using {[ORGS_TABLE, DEALS_TABLE]} as input, exported {len(categories)} time series in {EXPORT_DIR}"
    )
    logging.info(
        f"The time series correspond to the following technology categories: {categories}"
    )
