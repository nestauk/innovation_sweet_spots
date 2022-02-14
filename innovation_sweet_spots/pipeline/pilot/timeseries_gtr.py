"""
Script to generate time series from GtR data
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import import_config
import innovation_sweet_spots.analysis.analysis_utils as au
import innovation_sweet_spots.analysis.wrangling_utils as wu
from innovation_sweet_spots.getters.gtr import get_pilot_GtR_projects
import pandas as pd

# Parameters (tech categories to process, time series limits)
PARAMS = import_config("iss_pilot.yaml")
SPLIT = False
PERIOD = "month"

# Output files
EXPORT_DIR = (
    PROJECT_DIR / "outputs/finals/pilot_outputs/time_series/gtr_split" / PERIOD
    if SPLIT
    else PROJECT_DIR / "outputs/finals/pilot_outputs/time_series/gtr_not_split" / PERIOD
)


if __name__ == "__main__":
    # Load data
    gtr_docs = au.filter_years(
        df=get_pilot_GtR_projects(),
        date_col="start",
        keep_year_col=False,
        min_year=PARAMS["min_year"],
        max_year=PARAMS["max_year"],
    )

    # Technology categories to process
    categories = PARAMS["technology_categories"]

    if SPLIT:
        gtr_wrangler = wu.GtrWrangler()

    for category in categories:
        # Select rows relevant to the category
        category_gtr_docs = gtr_docs.query(f'tech_category=="{category}"').copy()

        if SPLIT:
            funding_data = gtr_wrangler.get_funding_data(category_gtr_docs)
            funding_data_split = gtr_wrangler.split_funding_data(funding_data)
            time_series_funding = au.gtr_get_all_timeseries_period(
                funding_data_split, PERIOD, PARAMS["min_year"], PARAMS["max_year"]
            )
        else:
            time_series_funding = au.gtr_get_all_timeseries_period(
                category_gtr_docs, PERIOD, PARAMS["min_year"], PARAMS["max_year"]
            )
        time_series_funding["tech_category"] = category
        # Make dir
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        # Export
        outfile_name = (
            f"Time_series_GtR_split_{category}_split_{PERIOD}.csv"
            if SPLIT
            else f"Time_series_GtR_{category}_{PERIOD}.csv"
        )
        time_series_funding.to_csv(EXPORT_DIR / outfile_name, index=False)

    logging.info(
        f"Generated and exported {len(categories)} time series in {EXPORT_DIR}"
    )
    logging.info(
        f"The time series correspond to the following technology categories: {categories}"
    )
