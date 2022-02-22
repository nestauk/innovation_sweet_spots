"""
Script to generate time series from GtR data
"""
import typer
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import import_config
import innovation_sweet_spots.analysis.analysis_utils as au
import innovation_sweet_spots.analysis.wrangling_utils as wu
from innovation_sweet_spots.getters.gtr import get_pilot_GtR_projects
import pandas as pd

# Parameters (tech categories to process, time series limits)
PARAMS = import_config("iss_pilot.yaml")


def gtr_timeseries(period: str, split: bool = False):
    """Loads, processes and saves Gateway to Research time
    series data

    Args:
        period: Period to group the data by, 'month', 'quarter' or 'year'
        split: If True, will evenly split research funding over the duration
            of the project. If False, will attribute research funding to the
            start of the project.
    """
    # Output files
    EXPORT_DIR = (
        PROJECT_DIR / "outputs/finals/pilot_outputs/time_series/gtr_split" / period
        if split
        else PROJECT_DIR
        / "outputs/finals/pilot_outputs/time_series/gtr_not_split"
        / period
    )
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

    if split:
        gtr_wrangler = wu.GtrWrangler()

    with typer.progressbar(categories) as progress:
        for category in progress:
            # Select rows relevant to the category
            category_gtr_docs = gtr_docs.query(f'tech_category=="{category}"').copy()

            if split:
                funding_data = gtr_wrangler.get_funding_data(category_gtr_docs)

                # Split monthly to group at higher level later to more accurately split funding
                funding_data_split_monthly = gtr_wrangler.split_funding_data(
                    funding_data, "month"
                )
                time_series_funding = au.gtr_get_all_timeseries_period(
                    funding_data_split_monthly,
                    period,
                    PARAMS["min_year"],
                    PARAMS["max_year"],
                )

                # Split by period (necessary for calculating median)
                funding_data_split_period = gtr_wrangler.split_funding_data(
                    funding_data, period
                )
                time_series_median = au.gtr_funding_median_per_period(
                    funding_data_split_period,
                    period,
                    PARAMS["min_year"],
                    PARAMS["max_year"],
                )

            else:
                time_series_funding = au.gtr_get_all_timeseries_period(
                    category_gtr_docs, period, PARAMS["min_year"], PARAMS["max_year"]
                )
                time_series_median = au.gtr_funding_median_per_period(
                    category_gtr_docs,
                    period,
                    PARAMS["min_year"],
                    PARAMS["max_year"],
                )

            time_series_funding_median = pd.merge(
                time_series_funding,
                time_series_median,
                how="outer",
                on="time_period",
            )

            time_series_funding_median["tech_category"] = category
            # Make dir
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)

            # Export
            outfile_name = (
                f"Time_series_GtR_split_{category}_split_{period}.csv"
                if split
                else f"Time_series_GtR_{category}_{period}.csv"
            )
            time_series_funding_median.to_csv(EXPORT_DIR / outfile_name, index=False)

        logging.info(
            f"Generated and exported {len(categories)} time series in {EXPORT_DIR}"
        )
        logging.info(
            f"The time series correspond to the following technology categories: {categories}"
        )


if __name__ == "__main__":
    typer.run(gtr_timeseries)
