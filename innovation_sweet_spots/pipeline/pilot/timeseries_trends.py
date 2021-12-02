"""
Script to calculate time series trends
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import import_config
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd

# Input files
DATA_DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/time_series"
# Output files
EXPORT_DIR = DATA_DIR
OUTFILE_NAME = "Trends_{}_{}.csv"
# Parameters (tech categories, time series limits)
SOURCES = ["GtR", "Crunchbase"]
PARAMS = import_config("iss_pilot.yaml")

if __name__ == "__main__":

    # Technology categories to process
    categories = PARAMS["technology_categories"]

    for source in SOURCES:
        for category in categories:
            # Import time series
            time_series = pd.read_csv(DATA_DIR / f"Time_series_{source}_{category}.csv")

            # Estimate growth trends
            investment_trends = au.estimate_magnitude_growth(
                time_series,
                year_start=PARAMS["trend_min_year"],
                year_end=PARAMS["trend_max_year"],
                window=PARAMS["window"],
            )
            investment_trends.to_csv(
                EXPORT_DIR / OUTFILE_NAME.format(source, category), index=False
            )

        logging.info(f"Exported {len(categories)} trends in {EXPORT_DIR}")
        logging.info(
            f"The trends correspond to {source} data, in the following technology categories: {categories}"
        )
