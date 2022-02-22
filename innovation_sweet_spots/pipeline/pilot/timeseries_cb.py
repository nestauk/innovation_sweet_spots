"""
Script to generate time series from Crunchbase data
"""
import typer
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import import_config
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd

# Input files
DATA_DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/"
ORGS_TABLE = "ISS_pilot_Crunchbase_companies.csv"
DEALS_TABLE = "ISS_pilot_Crunchbase_deals.csv"
# Parameters (tech categories to process, time series limits)
PARAMS = import_config("iss_pilot.yaml")
# Output files
OUTFILE_NAME = "Time_series_Crunchbase_{}_{}.csv"


def cb_timeseries(period: str):
    """Loads, processes and saves Crunchbase time series data

    Args:
        period: Period to group the data by, 'month', 'quarter' or 'year'
    """
    export_dir = DATA_DIR / "time_series/cb" / period
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    cb_orgs = au.filter_years(
        df=pd.read_csv(DATA_DIR / ORGS_TABLE),
        date_col="founded_on",
        keep_year_col=False,
        min_year=PARAMS["min_year"],
        max_year=PARAMS["max_year"],
    )

    cb_deals = au.filter_years(
        df=pd.read_csv(DATA_DIR / DEALS_TABLE),
        date_col="announced_on_date",
        keep_year_col=False,
        min_year=PARAMS["min_year"],
        max_year=PARAMS["max_year"],
    )

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
            period=period,
            min_year=PARAMS["min_year"],
            max_year=PARAMS["max_year"],
        )
        time_series_investment["tech_category"] = category

        # Export
        time_series_investment.to_csv(
            export_dir / OUTFILE_NAME.format(category, period), index=False
        )

    logging.info(
        f"Using {[ORGS_TABLE, DEALS_TABLE]} as input, exported {len(categories)} time series in {export_dir}"
    )
    logging.info(
        f"The time series correspond to the following technology categories: {categories}"
    )


if __name__ == "__main__":
    typer.run(cb_timeseries)
