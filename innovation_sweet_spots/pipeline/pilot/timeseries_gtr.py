"""
Script to generate time series from GtR data
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import import_config
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd

# Input files
DATA_DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/"
PROJECTS_TABLE = "ISS_pilot_GtR_projects.csv"
# Output files
EXPORT_DIR = DATA_DIR / "time_series/"
OUTFILE_NAME = "Time_series_GtR_{}.csv"
# Parameters (tech categories to process, time series limits)
PARAMS = import_config("iss_pilot.yaml")

if __name__ == "__main__":

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    gtr_docs = pd.read_csv(DATA_DIR / PROJECTS_TABLE)

    # Technology categories to process
    categories = PARAMS["technology_categories"]

    for category in categories:
        # Select rows relevant to the category
        category_gtr_docs = gtr_docs.query(f'tech_category=="{category}"').copy()

        # Calculate time series
        time_series_funding = au.gtr_get_all_timeseries(
            category_gtr_docs, min_year=PARAMS["min_year"], max_year=PARAMS["max_year"]
        )

        # Export
        time_series_funding.to_csv(
            EXPORT_DIR / OUTFILE_NAME.format(category), index=False
        )

    logging.info(
        f"Using {PROJECTS_TABLE} as input, generated and exported {len(categories)} time series in {EXPORT_DIR}"
    )
    logging.info(
        f"The time series correspond to the following technology categories: {categories}"
    )
