"""
Script to check time series data and trends calculations from Crunchbase data
"""
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd

DATA_DIR = PROJECT_DIR / "outputs/finals/pilot_outputs/"
PATH_TO_ORGS_TABLE = DATA_DIR / "ISS_pilot_Crunchbase_companies.csv"
PATH_TO_DEALS_TABLE = DATA_DIR / "ISS_pilot_Crunchbase_deals.csv"

EXPORT_DIR = DATA_DIR / "time_series/"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Time series limits
MIN_YEAR = 2007
MAX_YEAR = 2021
# Years for trend estimates
TREND_MIN_YEAR = 2016
TREND_MAX_YEAR = 2020
# Load data
cb_orgs = pd.read_csv(PATH_TO_ORGS_TABLE)
cb_deals = pd.read_csv(PATH_TO_DEALS_TABLE)
# Select technology category
category = "Low carbon heating"
# Select rows relevant to the category
category_cb_orgs = cb_orgs.query(f'tech_category=="{category}"').copy()
category_deals = cb_deals.query(f'tech_category=="{category}"').copy()
# Calculate time series
time_series_orgs_founded = au.cb_orgs_founded_per_year(
    category_cb_orgs, min_year=MIN_YEAR, max_year=MAX_YEAR
)
time_series_investment = au.cb_investments_per_year(
    category_deals, min_year=MIN_YEAR, max_year=MAX_YEAR
)
time_series_investment["no_of_orgs_founded"] = time_series_orgs_founded[
    "no_of_orgs_founded"
]
# Estimate growth trends
investment_trends = au.estimate_magnitude_growth(time_series_investment, 2016, 2020)
# Export
time_series_investment.to_csv(
    EXPORT_DIR / f"Time_series_Crunchbase_{category}.csv", index=False
)
investment_trends.to_csv(EXPORT_DIR / f"Trends_Crunchbase_{category}.csv", index=False)
