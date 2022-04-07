import pandas as pd
import numpy as np
from itertools import chain
from typing import Union
from ast import literal_eval


def convert_col_to_has_col(df: pd.DataFrame, col: str, drop: bool) -> pd.DataFrame:
    """Create new column based on another column
    where non nan values set to 1, nan values to 0

    Args:
        df: Dataframe to convert columns
        col: Column to convert
        drop: Drop original column after converting

    Returns:
        Dataframe with binarised new column
    """
    df[f"has_{col}"] = df[col].notnull().astype("int")
    if drop:
        return df.drop(columns=[col])
    return df


def convert_nan_list_to_no_industry_listed(industries: list) -> list:
    """If list of industries contains a nan item, return
    ['no_industry_listed'], else return the list unchanged
    """
    return ["no_industry_listed"] if np.nan in industries else industries


def ind_to_group(industries: list, industry_to_group_map: dict) -> set:
    """Turns list of industries into a set of wider category groups.
    If the industry does not map to a category group, the industry name will
    be used instead.

    Args:
        industries: List of industries that a company is in
        industry_to_group_map: Maps industries to wider category groups

    Returns:
        Flattened set of industries
    """
    groups = [
        [ind] if industry_to_group_map[ind] == [] else industry_to_group_map[ind]
        for ind in industries
    ]
    flattened = list(chain(*groups))
    return set(flattened)


def tech_cats_to_dummies(cb_data: pd.DataFrame) -> pd.DataFrame:
    """Convert tech_category column to dummy columns with values 0 or 1"""
    combine_tech_cats = (
        cb_data.groupby("id")
        .agg({"tech_category": lambda x: ", ".join(map(str, x))})
        .reset_index()
    )
    dummies = combine_tech_cats["tech_category"].str.get_dummies(sep=", ")
    id_dummies = combine_tech_cats.merge(
        dummies, left_index=True, right_index=True
    ).drop(columns=["tech_category"])

    return (
        cb_data.drop(columns=["tech_category"])
        .drop_duplicates()
        .merge(id_dummies, left_on="id", right_on="id")
    )


def add_industry_dummies(cb_orgs: pd.DataFrame) -> pd.DataFrame:
    """Adds dummy columns for industries"""
    industry_dummies = pd.get_dummies(cb_orgs["industry_clean"].explode()).sum(level=0)
    return cb_orgs.merge(industry_dummies, left_index=True, right_index=True)


def add_group_dummies(
    cb_orgs: pd.DataFrame, industry_to_group_map: dict
) -> pd.DataFrame:
    """Adds dummy columns for wider category groups

    Args:
        cb_orgs: Dataframe containing column industry_clean
        industry_to_group_map: Maps industries to wider category groups

    Returns:
        Dataframe with dummy columns for wider category groups added
    """
    cb_orgs["groups"] = cb_orgs["industry_clean"].apply(
        ind_to_group, args=(industry_to_group_map,)
    )
    group_dummies = pd.get_dummies(cb_orgs["groups"].explode()).sum(level=0)
    return cb_orgs.merge(group_dummies, left_index=True, right_index=True)


def dedupe_descriptions(cb_data: pd.DataFrame) -> pd.DataFrame:
    """For rows which are duplicates except for differences in their description
    column, drop the rows with no description or the shorter description"""
    dedupe_descs = (
        cb_data.groupby("id")
        .agg(
            {
                "long_description": lambda x: max(
                    [str(np.nan_to_num(desc, 0)) for desc in x], key=len
                )
            }
        )
        .reset_index()
    )
    return (
        cb_data.drop(columns="long_description")
        .drop_duplicates()
        .merge(dedupe_descs, left_on="id", right_on="id")
    )


def add_unstack_data(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_id: str,
    right_id: str,
    right_col_to_add: str,
    prefix: str,
) -> pd.DataFrame:
    """General function to add a column to the left dataframe
    from the right dataframe. If there are multiple matches in the
    right dataframe, additional columns will be created in the left dataframe
    for each match (numbered in chronological order)

    Args:
        left: Dataframe to have column added to
        right: Dataframe to have column added from
        left_id: Id from left dataframe to join on
        right_id: Id from right dataframe to join on
        right_col_to_add: Column to add from right dataframe
        prefix: Prefix to be added to new columns in left dataframe

    Returns:
        Left dataframe with columns added relating to specified column in right dataframe

    Example:
        left = pd.DataFrame({"id": ["1", "2"]})
        right = pd.DataFrame({
            "id": ["1", "1", "2"],
            "date": ["01/01/2020", "01/01/2021", "01/01/2022"]
            })
        add_unstack_data(
            left=left,
            right=right,
            left_id="id",
            right_id="id",
            right_col_to_add="date",
            prefix="right_date_"
        )
        ->
        pd.DataFrame({
            "id": ["1", "2"],
            "right_date_0": ["01/01/2020", "01/01/2022"],
            "right_date_1": ["01/01/2021", np.nan]
        })
    """
    unstacked = (
        left.merge(
            right[[right_id, right_col_to_add]],
            how="left",
            left_on=left_id,
            right_on=right_id,
        )
        .groupby(left_id)[right_col_to_add]
        .apply(lambda x: pd.Series(sorted(list(x))))
        .unstack()
        .add_prefix(prefix)
    )
    return left.merge(unstacked, how="left", left_on=left_id, right_on=left_id)


def add_acquired_on(
    cb_data: pd.DataFrame, acquired_on_data: pd.DataFrame
) -> pd.DataFrame:
    """Add columns for acquired on dates from the acquisitions data"""
    return add_unstack_data(
        left=cb_data,
        right=acquired_on_data,
        left_id="id",
        right_id="acquiree_id",
        right_col_to_add="acquired_on",
        prefix="acquired_on_",
    )


def add_went_public_on(cb_data: pd.DataFrame, ipo_data: pd.DataFrame) -> pd.DataFrame:
    """Add columns for went public on dates from the ipos data"""
    return add_unstack_data(
        left=cb_data,
        right=ipo_data,
        left_id="id",
        right_id="org_id",
        right_col_to_add="went_public_on",
        prefix="went_public_on_",
    )


def add_funding_round_ids(
    cb_data: pd.DataFrame, cb_funding_data: pd.DataFrame
) -> pd.DataFrame:
    """Add columns for each funding round id from the funding rounds data"""
    cb_fd = cb_funding_data[["org_id", "id"]].rename(columns={"id": "funding_round_id"})
    return add_unstack_data(
        left=cb_data,
        right=cb_fd,
        left_id="id",
        right_id="org_id",
        right_col_to_add="funding_round_id",
        prefix="funding_round_id_",
    )


def add_funding_round_dates(
    cb_data: pd.DataFrame, cb_funding_data: pd.DataFrame
) -> pd.DataFrame:
    """Add columns for each funding round date from the funding rounds data"""
    return add_unstack_data(
        left=cb_data,
        right=cb_funding_data,
        left_id="id",
        right_id="org_id",
        right_col_to_add="announced_on",
        prefix="funding_round_date_",
    )


def window_flag(
    cb_data: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    variable: str,
) -> pd.DataFrame:
    """Check that any columns relating to the specified variable have a date
    within the time window. Add a binary flag column, set to 1 if the variable
    columns are within the time window, else 0 if not

    Args:
        cb_data: Dataframe to check
        start_date: Window start date
        end_date: Window end date
        variable: Variable to find relevant columns. For example, if variable is
            "acquired_on", it will be used to find columns "acquired_on_1" and
            "acquired_on_2" etc

    Returns:
        Dataframe with additional column to indicate whether the specified variable
        has any values within the date window
    """
    cols_to_loop = cb_data.columns[cb_data.columns.str.contains(variable)]
    cb_data[f"{variable}_in_window"] = (
        pd.DataFrame(
            [
                pd.to_datetime(cb_data[col])
                .between(start_date, end_date, inclusive="both")
                .astype("int")
                for col in cols_to_loop
            ]
        )
        .transpose()
        .max(axis=1)
    )
    return cb_data


def future_flag(
    cb_data: pd.DataFrame, start_date: pd.DatetimeIndex, variable: str
) -> pd.DataFrame:
    """Adds a new column where the values are set to 1 if
    there are any of the instances of the specified variable
    that occur after the start date, if not, the values are set
    to 0.

    Args:
        cb_data: Dataframe to have column added to
        start_date: Date to check if variable occured after
        variable: Variable to check dates for

    Returns:
        Dataframe with additional flag indicating if variable
        happened in the future
    """
    cols_to_loop = cb_data.columns[cb_data.columns.str.contains(variable)]
    cb_data[f"future_{variable}"] = (
        pd.DataFrame(
            [
                (pd.to_datetime(cb_data[col]) > start_date).astype("int")
                for col in cols_to_loop
            ]
        )
        .transpose()
        .max(axis=1)
    )
    return cb_data


def add_n_funding_rounds_in_window(
    cb_data: pd.DataFrame, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex
) -> pd.DataFrame:
    """Adds column for how many funding rounds occured in the date window

    Args:
        cb_data: Dataframe to add number funding rounds data to
        start_date: Start date of the time window
        end_date: End date of the time window

    Returns:
        Dataframe with column added for how many funding rounds occured
        in the date window
    """
    cols_to_loop = cb_data.columns[cb_data.columns.str.contains("funding_round_date")]
    cb_data["n_funding_rounds"] = (
        pd.DataFrame(
            [
                pd.to_datetime(cb_data[col])
                .between(start_date, end_date, inclusive="both")
                .astype("int")
                for col in cols_to_loop
            ]
        )
        .transpose()
        .sum(axis=1)
    )
    return cb_data


def add_first_last_date_col_number(
    cb_data: pd.DataFrame,
    col_contains_string: str,
    last: bool,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    new_col: str,
) -> pd.DataFrame:
    """Add a new column with values of the column name number of the first or last
    dates across columns containing specified string

    Args:
        cb_data: Dataframe containing columns with date values
        col_contains_string: String to use to find columns containing that string
        last: True to find column name number relating to latest date,
            False to find column name number relating to first date
        start_date: Start date of the time window
        end_date: End date of the time window
        new_col: Name of the new column

    Returns:
        cb_data with a new column containing column name number relating
        to first or last date across specified columns
    """
    dates_in_window = keep_dates_in_window(
        cb_data, col_contains_string, start_date, end_date
    )
    cb_data[new_col] = (
        (
            dates_in_window.eq(dates_in_window.max(1), axis=0)
            if last
            else dates_in_window.eq(dates_in_window.min(1), axis=0)
        )
        .dot(dates_in_window.columns)
        .str.extract("(\d+)")
    )
    return cb_data


def add_last_funding_id_in_window(
    cb_data: pd.DataFrame,
) -> pd.DataFrame:
    """Adds column for last funding round id in window,
    needs col 'last_funding_round_in_window'"""
    last_funding_id_in_window = []
    for _, row in cb_data.iterrows():
        rnd = row["last_funding_round_in_window"]
        try:
            last_funding_id_in_window.append(row[f"funding_round_id_{rnd}"])
        except:
            last_funding_id_in_window.append(np.nan)
    cb_data["last_funding_id_in_window"] = last_funding_id_in_window
    return cb_data


def keep_dates_in_window(
    cb_data: pd.DataFrame,
    col_contains_string: str,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return columns containing specified string, keep values with
    dates within the start and end date provided

    Args:
        cb_data: Dataframe containing columns with date values
        col_contains_string: String to use to find columns containing that string
        start_date: Start date of the time window
        end_date: End date of the time window

    Returns:
        Dataframe with columns containing specified string, only with values
            within the start and end date provided
    """
    cols_to_loop = cb_data.columns[cb_data.columns.str.contains(col_contains_string)]
    return cb_data[cols_to_loop][
        pd.DataFrame(
            [
                pd.to_datetime(cb_data[col]).between(
                    start_date, end_date, inclusive="both"
                )
                for col in cols_to_loop
            ]
        ).transpose()
    ].astype("datetime64[ns]")


def add_first_last_date(
    cb_data: pd.DataFrame,
    col_contains_string: str,
    last: bool,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    new_col: str,
) -> pd.DataFrame:
    """Add a new column for first or last dates across columns
    containing specified string

    Args:
        cb_data: Dataframe to add number of months since last investment to
        col_contains_string: String to use to find columns containing that string
        last: True to find latest date, False to find first date
        start_date: Start date of the time window
        end_date: End date of the time window
        new_col: Name of new column

    Returns:
        cb_data with a new column containing first or last date across
            specified columns
    """
    dates_in_window = keep_dates_in_window(
        cb_data, col_contains_string, start_date, end_date
    )
    cb_data[new_col] = (
        dates_in_window.max(axis=1) if last else dates_in_window.min(axis=1)
    )
    return cb_data


def n_months_delta(
    date1: Union[str, pd.Series], date2: Union[str, pd.Series]
) -> Union[int, pd.Series]:
    """Returns number of months between two dates, returns -1 for NaT values"""
    return (
        ((pd.to_datetime(date1) - pd.to_datetime(date2)) / np.timedelta64(1, "M"))
        .fillna(-1)
        .astype(int)
    )


def add_n_months_since_last_investment_in_window(
    cb_data: pd.DataFrame, end_date: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add column for number of months since last investment in the time window

    Args:
        cb_data: Dataframe to add number of months since last investment to,
            must contain column for 'latest_funding_date_in_window'
        end_date: End date of the time window

    Returns:
        Dataframe with column added for number of months since last investment
        in the time window
    """
    cb_data["n_months_since_last_investment"] = n_months_delta(
        end_date, cb_data["latest_funding_date_in_window"]
    )
    return cb_data


def add_last_investment_round_info(
    cb_data: pd.DataFrame, cb_funding_rounds: pd.DataFrame
) -> pd.DataFrame:
    """Add variables relating to lastest investment round

    Args:
        cb_data: Dataframe containing column
            for 'last_funding_id_in_window'
        cb_funding_rounds: DataFrame containing funding round
            ids and associated funding round information

    Returns:
        cb_data with additional variables relating to latest investment round
    """
    return (
        cb_data.merge(
            right=cb_funding_rounds[["id", "investment_type", "raised_amount_gbp"]],
            how="left",
            left_on="last_funding_id_in_window",
            right_on="id",
        )
        .drop(columns="id_y")
        .rename(
            columns={
                "investment_type": "last_investment_round_type",
                "raised_amount_gbp": "last_investment_round_gbp",
                "id_x": "id",
            }
        )
        .fillna(
            {
                "last_investment_round_type": "no_last_round",
                "last_investment_round_gbp": -1,
            }
        )
    )


def add_n_months_before_first_investment_in_window(
    cb_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add column for number of months before first investment in the time window

    Args:
        cb_data: Dataframe to add number of months before first investment to,
            must contain column for 'first_funding_date_in_window' and 'founded_on'

    Returns:
        Dataframe with column added for number of months before first investment
        in the time window
    """
    cb_data["n_months_before_first_investment"] = n_months_delta(
        cb_data["first_funding_date_in_window"], cb_data["founded_on"]
    )
    return cb_data


def add_n_unique_investors_total(
    cb_data: pd.DataFrame,
    cb_funding_rounds: pd.DataFrame,
    cb_investments: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add column for number of unique investors total

    Args:
        cb_data: Dataframe to add column for number of unique investors total to
        cb_funding_rounds: Dataframe containing information about crunchbase
            funding rounds
        cb_investments: Dataframe containing information about crunchbase investments
        start_date: Start date of the time window
        end_date: End date of the time window

    Returns:
        cb_data with additional column for number of unique investors total

    """
    funding_rounds_in_window = (
        cb_funding_rounds.astype({"announced_on": "datetime64[ns]"})
        .query(f"'{start_date}' <= announced_on <= '{end_date}'")[["id", "org_id"]]
        .rename(columns={"id": "funding_round_id"})
    )
    n_unique_investors = (
        funding_rounds_in_window.merge(
            right=cb_investments[["funding_round_id", "investor_id"]],
            how="left",
            left_on="funding_round_id",
            right_on="funding_round_id",
        )
        .drop(columns="funding_round_id")
        .drop_duplicates()
        .groupby("org_id")["investor_id"]
        .agg("count")
        .reset_index(name="n_unique_investors_total")
    )
    return cb_data.merge(
        n_unique_investors, left_on="id", right_on="org_id", how="left"
    ).fillna({"n_unique_investors_total": -1})


def add_n_unique_investors_last_round(
    cb_data: pd.DataFrame, cb_investments: pd.DataFrame
) -> pd.DataFrame:
    """Add column for number of unique investors in the last funding round"""
    n_unique_investors_last_round = (
        cb_data[["id", "last_funding_id_in_window"]]
        .dropna(subset=["last_funding_id_in_window"])
        .merge(
            right=cb_investments[["funding_round_id", "investor_id"]],
            how="left",
            left_on="last_funding_id_in_window",
            right_on="funding_round_id",
        )[["id", "investor_id"]]
        .drop_duplicates()
        .groupby("id")["investor_id"]
        .agg("count")
        .reset_index(name="n_unique_investors_last_round")
    )
    return cb_data.merge(
        n_unique_investors_last_round, left_on="id", right_on="id", how="left"
    ).fillna({"n_unique_investors_last_round": -1})


def add_n_months_since_founded(
    cb_data: pd.DataFrame, end_date: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add number of months since company was founded (where the
    end of the time window date is the 'present' date)

    Args:
        cb_data: Dataframe to add the number of months since founded column to
        end_date: End date of the time window to act as the 'present' date

    Returns:
        Dataframe with additional column for number of month since the company
        was founded
    """
    cb_data["n_months_since_founded"] = n_months_delta(end_date, cb_data["founded_on"])
    return cb_data


def total_investment(
    cb_funding_rounds: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Produce a dataframe containing org_id and total_investment_amount_gbp
    for within the specified date range"""
    return (
        cb_funding_rounds.astype({"announced_on": "datetime64[ns]"})
        .query(f"'{start_date}' <= announced_on <= '{end_date}'")
        .groupby("org_id")["raised_amount_gbp"]
        .agg("sum")
        .reset_index(name="total_investment_amount_gbp")
    )


def add_total_investment(
    cb_data: pd.DataFrame,
    cb_funding_rounds: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> pd.DataFrame:
    "Add column to cb_data for total gbp investment in the specified date range"
    total_invest = total_investment(cb_funding_rounds, start_date, end_date)
    return cb_data.merge(
        right=total_invest, left_on="id", right_on="org_id", how="left", validate="1:1"
    ).fillna({"total_investment_amount_gbp": -1})


def drop_multi_cols(
    cb_data: pd.DataFrame, cols_to_drop_str_containing: list
) -> pd.DataFrame:
    """Drop columns containing strings in list 'cols_to_drop_str_containing' provided"""
    return cb_data.drop(
        columns=cb_data.columns[
            cb_data.columns.str.contains("|".join(cols_to_drop_str_containing))
        ]
    )


def add_clean_job_title(cb_people: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned job title column to crunchbase people dataset"""
    cb_people["clean_job_title"] = cb_people.featured_job_title.str.lower()
    return cb_people


def add_is_founder(cb_people: pd.DataFrame) -> pd.DataFrame:
    """Add is_founder column to crunchbase people dataset"""
    cb_people["is_founder"] = (
        cb_people.clean_job_title.str.contains(
            "founde"
        )  # not a typo but quite a few founde typos in dataset
        .fillna(-1)
        .astype("int")
        .replace(-1, np.nan)
    )
    return cb_people


def add_is_gender(cb_people: pd.DataFrame, gender: str) -> pd.DataFrame:
    """Add is_male_founder or is_female_founder to crunchbase people dataset"""
    cb_people[f"is_{gender}_founder"] = np.where(
        ((cb_people.is_founder == 1) & (cb_people.gender == gender)), 1, 0
    )
    return cb_people


def person_id_degree_count(cb_degrees: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with cols for person_id and degree_count"""
    return (
        cb_degrees.groupby("person_id")["id"]
        .agg("count")
        .reset_index(name="degree_count")
    )


def add_founders(cb_data: pd.DataFrame, org_id_founders: pd.DataFrame) -> pd.DataFrame:
    """Adds columns relating to the founders of the companies

    Args:
        cb_data: dataframe to add additional founders columns to
        org_id_founders: dataframe to be merged, containing founders columns

    Returns:
        cb_data with additional columns for:
            - founder_count
            - female_founder_count
            - male_founder_count
            - founder_max_degrees
            - founder_mean_degrees
    """
    return cb_data.merge(
        org_id_founders, how="left", left_on="id", right_on="org_id"
    ).fillna(
        {
            "founder_count": -1,
            "female_founder_count": -1,
            "male_founder_count": -1,
            "founder_max_degrees": -1,
            "founder_mean_degrees": -1,
        }
    )


def add_grants(cb_data: pd.DataFrame, grants: pd.DataFrame) -> pd.DataFrame:
    """Add grants related features to the dataset

    Args:
        cb_data: dataframe to add additional grants related columns to
        grants: dataframe containing grants features

    Returns:
        cb_data dataframe with additional columns for:
            - total_grant_amount_gbp
            - n_grants
            - first_grant_date
            - last_grant_date
            - has_received_ukri_grant
            - has_received_grant
            - last_grant_amount_gbp

    """
    return cb_data.merge(
        right=grants, how="left", left_on="id", right_on="cb_org_id"
    ).fillna(
        {
            "total_grant_amount_gbp": -1,
            "n_grants": -1,
            "has_received_ukri_grant": -1,
            "has_received_grant": -1,
            "last_grant_amount_gbp": -1,
        }
    )


def add_n_months_since_last_grant(
    cb_data: pd.DataFrame, end_date: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add column for number of months since last grant

    Args:
        cb_data: Dataframe to add number of months since last grant to,
            must contain column for 'last_grant_date'
        end_date: End date of the time window

    Returns:
        Dataframe with column added for number of months since last grant
    """
    cb_data["n_months_since_last_grant"] = n_months_delta(
        end_date, cb_data["last_grant_date"]
    )
    return cb_data


def add_n_months_before_first_grant(
    cb_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add column for number of months before first grant

    Args:
        cb_data: Dataframe to add number of months before first grant,
            must contain column for 'first_grant_date' and 'founded_on'

    Returns:
        Dataframe with column added for number of months before first grant
    """
    cb_data["n_months_before_first_grant"] = n_months_delta(
        cb_data["first_grant_date"], cb_data["founded_on"]
    )
    return cb_data


def remove_gtr_grants_from_cb_grants(
    cb_grants_gbp: pd.DataFrame,
    cb_investments: pd.DataFrame,
    ukri_grant_providers_to_filter: list,
) -> pd.DataFrame:
    """Remove grants that are from UKRI from crunchbase grants
    to avoid having a grant picked from both crunchbase and
    gateway to research

    Args:
        cb_grants_gbp: Crunchbase grants dataframe
        cb_investments: Crunchbase investments dataframe
        ukri_grant_providers_to_filter: UKRI councils to find grants
            from and filter out

    Returns:
        cb_grants_gbp but with grants from UKRI removed
    """
    cb_grants_gbp_w_investments = cb_grants_gbp.merge(
        right=cb_investments,
        how="left",
        left_on="funding_round_id",
        right_on="funding_round_id",
    )
    cb_grants_gbp_w_investments_in_gtr = cb_grants_gbp_w_investments[
        cb_grants_gbp_w_investments.investor_name.str.contains(
            ukri_grant_providers_to_filter
        ).fillna(False)
    ]
    gtr_funding_rounds_to_drop = list(
        cb_grants_gbp_w_investments_in_gtr.funding_round_id.drop_duplicates().values
    )
    return cb_grants_gbp[
        ~cb_grants_gbp.funding_round_id.isin(gtr_funding_rounds_to_drop)
    ].reset_index(drop=True)


def standardise_cb_grants(cb_grants_without_gtr_grants: pd.DataFrame) -> pd.DataFrame:
    """Standard crunchbase grants data so that it can be
    combined with gtr grants data"""
    return (
        cb_grants_without_gtr_grants.rename(
            columns={
                "funding_round_id": "grant_id",
                "announced_on": "grant_date",
                "raised_amount_gbp": "grant_amount_gbp",
                "org_id": "cb_org_id",
            }
        )
        .assign(ukri_grant=0)
        .drop(columns="org_name")
    )


def explode_cb_gtr_lookup(cb_gtr_lookup: pd.DataFrame) -> pd.DataFrame:
    """Explode crunchase to gtr organisation lookup"""
    # Convert gtr_org_ids type to string
    cb_gtr_lookup["gtr_org_ids"] = cb_gtr_lookup["gtr_org_ids"].apply(literal_eval)
    # Explode cb to gtr lookup
    return cb_gtr_lookup.explode("gtr_org_ids").reset_index(drop=True)


def add_gtr_project_to_cb_gtr_lookup(
    cb_gtr_org_lookup: pd.DataFrame, gtr_grants: pd.DataFrame
) -> pd.DataFrame:
    """Add gtr project information to cb_gtr_org_lookup"""
    return (
        cb_gtr_org_lookup.merge(
            right=gtr_grants, left_on="gtr_org_ids", right_on="gtr_org_id", how="left"
        )
        .dropna(subset=["gtr_org_id"])
        .drop(columns="gtr_org_ids")
        .drop_duplicates(subset=["project_id", "cb_org_id"])
        .reset_index(drop=True)
    )


def standardise_gtr_grants(cb_gtr_lookup_projects):
    """Standard gtr grants data so that it can be
    combined with crunchbase grants data"""
    return (
        cb_gtr_lookup_projects.rename(
            columns={
                "project_id": "grant_id",
                "fund_start": "grant_date",
                "amount": "grant_amount_gbp",
            }
        )
        .assign(ukri_grant=1)
        .drop(columns="gtr_org_id")
    )


def join_cb_gtr_grants(
    cb_grants_standardised: pd.DataFrame, gtr_grants_standardised: pd.DataFrame
) -> pd.DataFrame:
    """Join crunchbase and gtr grants together"""
    return pd.concat([cb_grants_standardised, gtr_grants_standardised]).reset_index(
        drop=True
    )


def grants_features(combined_cb_gtr_grants: pd.DataFrame) -> pd.DataFrame:
    """Create groupby grants features for each crunchbase organisation id"""
    return (
        combined_cb_gtr_grants.fillna({"grant_amount_gbp": 0})
        .groupby("cb_org_id")
        .agg(
            total_grant_amount_gbp=("grant_amount_gbp", "sum"),
            n_grants=("grant_id", "count"),
            first_grant_date=("grant_date", "min"),
            last_grant_date=("grant_date", "max"),
            has_received_ukri_grant=("ukri_grant", "max"),
        )
        .assign(has_received_grant=1)
        .reset_index()
    )


def last_grant_amount(combined_cb_gtr_grants: pd.DataFrame) -> pd.DataFrame:
    """Calculate last grant amount received for each crunchbase organisation id"""
    return (
        combined_cb_gtr_grants.fillna({"grant_amount_gbp": 0})
        .sort_values("grant_date", ascending=False)
        .groupby("cb_org_id")
        .head(1)
        .rename(
            columns={
                "grant_date": "last_grant_date",
                "grant_amount_gbp": "last_grant_amount_gbp",
            }
        )[["cb_org_id", "last_grant_amount_gbp"]]
    )


def gtr_projects_with_lead_orgs(
    gtr_wrangler: classmethod,
    gtr_lead_orgs_to_project_id_lookup: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Use GtrWrangler to find gtr projects within the time window and
    add funding and lead organisation information.

    Note that this creates a small amount of assignments of a project to
    more than one 'lead' gtr organisation

    Args:
        gtr_wrangler: class which can be used to get gtr project data
            and add funding information
        gtr_lead_orgs_to_project_id_lookup: Lookup that can be used to
            link gtr organisations and projects
        start_date: Window start date
        end_date: Window end date

    Returns:
        Dataframe with columns for:
            - project_id
            - fund_start
            - amount
            - gtr_org_id
    """
    return (
        gtr_wrangler.get_funding_data(gtr_wrangler.gtr_projects)
        .query(f"'{start_date}' <= fund_start <= '{end_date}'")
        .reset_index(drop=True)
        .merge(
            right=gtr_lead_orgs_to_project_id_lookup,
            left_on="project_id",
            right_on="project_id",
            how="left",
        )[["project_id", "fund_start", "amount", "gtr_org_id"]]
    )
