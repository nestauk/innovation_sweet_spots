"""
Script to create csv dataset using Crunchbase and Gateway to Research data that can be
used for predicting future investment sucess for companies.

Run the following command in the terminal to see the options for creating the dataset:
python innovation_sweet_spots/pipeline/pilot/investment_predictions/create_dataset/create_dataset.py --help

On an M1 macbook it takes ~7 mins to run on the full dataset and ~1 min 30 secs to run in test mode.
"""
import typer
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters import crunchbase
from innovation_sweet_spots.getters.crunchbase import (
    get_crunchbase_funding_rounds,
    get_crunchbase_ipos,
    get_crunchbase_acquisitions,
    get_crunchbase_orgs,
    get_crunchbase_investments,
    get_crunchbase_people,
    get_crunchbase_degrees,
)
from innovation_sweet_spots.pipeline.pilot.investment_predictions.create_dataset import (
    utils,
)
from innovation_sweet_spots.pipeline.pilot.investment_predictions.create_dataset import (
    utils,
)
import pandas as pd
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

# Adjust the Crunchbase data snapshot path (ideally, should adapt the code to accommodate the newest snapshot)
crunchbase.CB_PATH = crunchbase.CB_PATH.parents[0] / "cb_2021"

KEEP_COLS = [
    "id",
    "name",
    "legal_name",
    "long_description",
    "location_id",
    "industry_clean",
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

DROP_MULTI_COLS = [
    "funding_round_date",
    "funding_round_id",
    "acquired_on",
    "went_public_on",
]

DROP_COLS = [
    "founded_on",
    "closed_on",
    "industry_clean",
    "groups",
    "first_funding_date_in_window",
    "last_funding_round_in_window",
    "latest_funding_date_in_window",
    "last_funding_id_in_window",
    "org_id_x",
    "org_id_y",
    "org_id",
    "last_funding_id_in_window",
]


def create_dataset(
    window_start_date: str = "01/01/2010",
    window_end_date: str = "01/01/2018",
    industries_or_groups: str = "groups",
    test: bool = False,
):
    """Loads crunchbase data, processes and saves dataset which can be used
    to predict future investment success for companies

    Args:
        window_start_date: Start date for window that simulates the
            evaluation period for assessing companies. Defaults to "01/01/2014".
        window_end_date: End date for window that simulates the evaluation period
            for assessing companies. Defaults to "01/01/2018".
        industries_or_groups: 'industries' to have a column to indicate which
            industries the company is in or 'groups' to have a column to indicate
            which wider industry group the company is in.
        test: If set to True, reduces crunchbase orgs to 5000 records. Set to
            True to quickly check functionality.
    """
    # Date information
    window_start_date = pd.to_datetime(window_start_date)
    window_end_date = pd.to_datetime(window_end_date)
    success_start_date = window_end_date

    # Create a CrunchbaseWrangler
    cb_wrangler = CrunchbaseWrangler()

    # Load datasets
    cb_orgs = (
        get_crunchbase_orgs()
        .query("country_code == 'GBR'")
        .query(f"'{window_start_date}' <= founded_on")
        .reset_index()
    )
    if test:
        cb_orgs = cb_orgs.head(2000)
    cb_acquisitions = get_crunchbase_acquisitions()
    cb_ipos = get_crunchbase_ipos()
    cb_funding_rounds_usd = get_crunchbase_funding_rounds().query(
        f"'{window_start_date}' <= announced_on"
    )
    cb_investments = get_crunchbase_investments()
    cb_people = get_crunchbase_people()
    cb_degrees = get_crunchbase_degrees()

    # Convert funding amounts to GBP
    cb_funding_rounds_gbp = cb_wrangler.convert_deal_currency_to_gbp(
        funding=cb_funding_rounds_usd.astype({"announced_on": "datetime64[ns]"}),
        date_column="announced_on",
        amount_column="raised_amount",
        usd_amount_column="raised_amount_usd",
        converted_column="raised_amount_gbp",
    )

    # Dedupe descriptions
    cb_orgs = cb_orgs.pipe(utils.dedupe_descriptions)

    # Create dict of industry to wider category groupings
    industry_to_group_map = cb_wrangler.industry_to_group
    industry_to_group_map["no_industry_listed"] = []

    # Add industries
    inds = cb_wrangler.get_company_industries(cb_orgs, return_lists=True)
    # Rename nan industries
    inds["industry_clean"] = inds["industry"].apply(
        utils.convert_nan_list_to_no_industry_listed
    )
    # Merge industries into cb_orgs
    cb_orgs = cb_orgs.merge(inds[["id", "industry_clean"]], left_on="id", right_on="id")

    # Binarise columns
    for col in BINARISE_COLS:
        cb_orgs = utils.convert_col_to_has_col(df=cb_orgs, col=col, drop=True)

    # Remove columns that are not needed
    cb_orgs = cb_orgs[KEEP_COLS]

    # Add dummy columns for industry information
    if industries_or_groups is "industries":
        cb_orgs = cb_orgs.pipe(utils.add_industry_dummies)
    if industries_or_groups is "groups":
        cb_orgs = cb_orgs.pipe(utils.add_group_dummies, industry_to_group_map)

    # Add founder info to people info
    cb_people = (
        cb_people.pipe(utils.add_clean_job_title)
        .pipe(utils.add_is_founder)
        .pipe(utils.add_is_gender, gender="male")
        .dropna(subset=["featured_job_organization_id"])
        .rename(
            columns={
                "id": "person_id",
                "featured_job_organization_id": "org_id",
            }
        )
        .reset_index(drop=True)
    )

    # Create dataframe for person id and degree count
    person_degree_count = utils.person_id_degree_count(cb_degrees)

    # Create dataframe for founders with gender and degree data
    cb_founders = cb_people.query("is_founder == 1").merge(
        person_degree_count, how="left", left_on="person_id", right_on="person_id"
    )

    # Create dataframe for org_id with grouped founders data
    org_id_founders = (
        cb_founders.groupby("org_id").agg(
            founder_count=("is_founder", "sum"),
            male_founder_percentage=("is_male_founder", "mean"),
            founder_max_degrees=("degree_count", "max"),
            founder_mean_degrees=("degree_count", "mean"),
        )
    ).reset_index()

    (
        # Add flag for founded on and filter out companies with 0 flag
        cb_orgs.pipe(
            utils.window_flag,
            start_date=window_start_date,
            end_date=window_end_date,
            variable="founded_on",
        )
        .query("founded_on_in_window == 1")
        # Add additional variables
        .pipe(utils.add_acquired_on, cb_acquisitions)
        .pipe(utils.add_went_public_on, cb_ipos)
        .pipe(utils.add_funding_round_ids, cb_funding_rounds_gbp)
        .pipe(utils.add_funding_round_dates, cb_funding_rounds_gbp)
        # Add flags for company acquired on and went public in time window
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
        # Filter companies not acquired or went public in the time window
        .query("acquired_on_in_window == 0 & went_public_on_in_window == 0")
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
        # Add col for latest funding date in window
        .pipe(
            utils.add_first_last_date,
            "funding_round_date",
            True,
            window_start_date,
            window_end_date,
            "latest_funding_date_in_window",
        )
        # Add col for first funding date in window
        .pipe(
            utils.add_first_last_date,
            "funding_round_date",
            False,
            window_start_date,
            window_end_date,
            "first_funding_date_in_window",
        )
        # Add col for last funding round number in window
        .pipe(
            utils.add_first_last_date_col_number,
            col_contains_string="funding_round_date",
            last=True,
            start_date=window_start_date,
            end_date=window_end_date,
            new_col="last_funding_round_in_window",
        )
        # Add col for last funding id in window
        .pipe(utils.add_last_funding_id_in_window)
        # Add col for last_investment_round_type and last_investment_round_usd
        .pipe(utils.add_last_investment_round_info, cb_funding_rounds_gbp)
        # Add col for number of months before first investment
        .pipe(utils.add_n_months_before_first_investment_in_window)
        # Add col for total investment received
        .pipe(
            utils.add_total_investment,
            cb_funding_rounds_gbp,
            window_start_date,
            window_end_date,
        )
        # Add col for number of funding rounds
        .pipe(
            utils.add_n_funding_rounds_in_window,
            start_date=window_start_date,
            end_date=window_end_date,
        )
        # Add col for number of months since last investment
        .pipe(
            utils.add_n_months_since_last_investment_in_window,
            end_date=window_end_date,
        )
        # Add col for number of months since founded
        .pipe(utils.add_n_months_since_founded, end_date=window_end_date)
        # Add col for number of unique investors in the last funding round
        .pipe(utils.add_n_unique_investors_last_round, cb_investments=cb_investments)
        # Add col for number of unique investors total
        .pipe(
            utils.add_n_unique_investors_total,
            cb_funding_rounds=cb_funding_rounds_gbp,
            cb_investments=cb_investments,
            start_date=window_start_date,
            end_date=window_end_date,
        )
        # Add cols for founders information
        .pipe(utils.add_founders, org_id_founders)
        # Drop columns
        .pipe(
            utils.drop_multi_cols,
            cols_to_drop_str_containing=DROP_MULTI_COLS,
        )
        .drop(columns=DROP_COLS)
        .reset_index(drop=True)
        # Save to csv
        .to_csv(
            PROJECT_DIR
            / "outputs/finals/pilot_outputs/"
            / f"investment_predictions/company_data_window_{str(window_start_date).split(' ')[0]}-{str(window_end_date).split(' ')[0]}.csv"
        )
    )


if __name__ == "__main__":
    typer.run(create_dataset)
