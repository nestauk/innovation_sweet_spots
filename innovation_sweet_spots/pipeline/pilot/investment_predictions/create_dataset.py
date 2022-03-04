"""
Script to create csv using Crunchbase data that can be
used for predicting future investment sucess for companies
"""
import typer
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters.crunchbase import (
    get_crunchbase_funding_rounds,
    get_crunchbase_ipos,
    get_crunchbase_acquisitions,
)
import utils
import pandas as pd

KEEP_COLS = [
    "id",
    "name",
    "legal_name",
    "description",
    "location_id",
    "tech_category",
    "has_email",
    "has_phone",
    "has_facebook_url",
    "has_twitter_url",
    "has_homepage_url",
    "has_linkedin_url",
    "founded_on",
    "closed_on",
]

SUCCESS_COLS = [
    "future_funding_round_date",
    "future_acquired_on",
    "future_went_public_on",
]

BINARISE_COLS = [
    "twitter_url",
    "email",
    "phone",
    "facebook_url",
    "homepage_url",
    "linkedin_url",
]


def create_dataset(
    window_start_date: str = "01/01/2014", window_end_date: str = "01/01/2018"
):
    """Loads crunchbase data, processes and saves dataset which can be used
    to predict future investment success for companies

    Args:
        window_start_date: Start date for window that simulates the
            evaluation period for assessing companies. Defaults to "01/01/2014".
        window_end_date: End date for window that simulates the evaluation period
            for assessing companies. Defaults to "01/01/2018".
    """
    # Date information
    window_start_date = pd.to_datetime(window_start_date)
    window_end_date = pd.to_datetime(window_end_date)
    success_start_date = window_end_date
    pilot_outputs = PROJECT_DIR / "outputs/finals/pilot_outputs/"
    # Load datasets
    cb_orgs = pd.read_csv(pilot_outputs / "ISS_pilot_Crunchbase_companies.csv")
    cb_acquisitions = get_crunchbase_acquisitions()
    cb_ipos = get_crunchbase_ipos()
    cb_funding_rounds = get_crunchbase_funding_rounds()

    # Binarise columns
    for col in BINARISE_COLS:
        cb_orgs = utils.convert_col_to_has_col(df=cb_orgs, col=col, drop=True)

    (
        # Select relevant cols
        cb_orgs[KEEP_COLS]
        # Add additional variables
        .pipe(utils.tech_cats_to_dummies)
        .pipe(utils.dedupe_descriptions)
        .pipe(utils.add_acquired_on, cb_acquisitions)
        .pipe(utils.add_went_public_on, cb_ipos)
        .pipe(utils.add_funding_round_ids, cb_funding_rounds)
        .pipe(utils.add_funding_round_dates, cb_funding_rounds)
        # Add flags for company founded/acquired/went public in time window
        .pipe(
            utils.window_flag,
            start_date=window_start_date,
            end_date=window_end_date,
            variable="founded_on",
        )
        .pipe(
            utils.window_flag,
            start_date=window_start_date,
            end_date=window_end_date,
            variable="acquired_on",
        )
        .pipe(
            utils.window_flag,
            start_date=window_start_date,
            end_date=window_end_date,
            variable="went_public_on",
        )
        # Filter companies founded in the time window but not acquired or went public
        .query(
            "founded_on_in_window == 1 & acquired_on_in_window == 0 & went_public_on_in_window == 0"
        )
        .drop(
            columns=[
                "founded_on_in_window",
                "acquired_on_in_window",
                "went_public_on_in_window",
            ]
        )
        # Add flags for each measure of success
        .pipe(
            utils.future_flag,
            start_date=success_start_date,
            variable="funding_round_date",
        )
        .pipe(utils.future_flag, start_date=success_start_date, variable="acquired_on")
        .pipe(
            utils.future_flag, start_date=success_start_date, variable="went_public_on"
        )
        # Create future_success variable which is set to 1 if one of the above flags is 1
        .assign(future_success=lambda x: x[SUCCESS_COLS].max(axis=1))
        .drop(columns=SUCCESS_COLS)
        .pipe(
            utils.add_n_funding_rounds_in_window,
            start_date=window_start_date,
            end_date=window_end_date,
        )
        .pipe(
            utils.add_n_months_since_last_investment_in_window,
            start_date=window_start_date,
            end_date=window_end_date,
        )
        .pipe(utils.add_n_months_since_founded, end_date=window_end_date)
        # Drop columns
        .pipe(
            utils.drop_multi_cols,
            cols_to_drop_str_containing=[
                "funding_round_date",
                "funding_round_id",
                "acquired_on",
                "went_public_on",
            ],
        )
        .drop(columns=["founded_on", "closed_on"])
        .reset_index(drop=True)
        # Save to csv
        .to_csv(
            pilot_outputs
            / f"investment_predictions/company_data_window_{str(window_start_date).split(' ')[0]}-{str(window_end_date).split(' ')[0]}.csv"
        )
    )


if __name__ == "__main__":
    typer.run(create_dataset)
