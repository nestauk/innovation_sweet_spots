"""
Utils for doing data analysis
"""
from innovation_sweet_spots import logging
import pandas as pd
import numpy as np
from innovation_sweet_spots.analysis.wrangling_utils import check_valid


def impute_empty_periods(
    df_time_period: pd.DataFrame,
    time_period_col: str,
    period: str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """
    Imputes zero values for time periods without data

    Args:
        df_time_period: A dataframe with a column containing time period data
        time_period_col: Column containing time period data
        period: Time period that the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with imputed 0s for time periods with no data
    """
    max_year_data = np.nan_to_num(df_time_period[time_period_col].max().year)
    max_year = max(max_year_data, max_year)
    full_period_range = (
        pd.period_range(
            f"01/01/{min_year}",
            f"31/12/{max_year}",
            freq=period,
        )
        .to_timestamp()
        .to_frame(index=False, name=time_period_col)
        .reset_index(drop=True)
    )
    return full_period_range.merge(df_time_period, "left").fillna(0)


### GtR specific utils


def gtr_deduplicate_projects(gtr_docs: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicates projects that have the same title and description. This can be used
    to report the overall funding amount of the whole project when it has
    received funding in separate installments across different years.

    Args:
        gtr_docs: A dataframe with columns with GtR project 'title', 'description',
            'amount' (research funding) and other data

    Returns:
        A dataframe where projects with the exact same title and description have
        been merged and their funding has been summed up
    """
    gtr_docs_summed_amounts = (
        gtr_docs.groupby(["title", "description"])
        .agg(amount=("amount", "sum"))
        .reset_index()
    )
    # Add the summed up amounts to the project and keep the earliest instance
    # of the duplicates
    return (
        gtr_docs.drop("amount", axis=1)
        .merge(gtr_docs_summed_amounts, on=["title", "description"], how="left")
        .sort_values("start")
        .drop_duplicates(["title", "description"], keep="first")
        .reset_index(drop=True)
        # Restore previous column order
    )[gtr_docs.columns]


def gtr_funding_per_period(
    gtr_docs: pd.DataFrame, period: str, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Given a table with projects and their funding, return an aggregation by period

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        period: Time period to group the data by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'period' - time period
            'no_of_projects' - number of new projects in a given period,
            'amount_total' - total amount of research funding in a given period
    """
    # Convert project start dates to time period
    gtr_docs = (
        gtr_docs.copy()
        .astype({"start": "datetime64[ns]"})
        .assign(time_period=lambda x: x.start.dt.strftime("%Y-%m-%d"))
        .astype({"time_period": "datetime64[ns]"})
    )
    # Group by time period
    grouped = gtr_docs.groupby(gtr_docs["time_period"].dt.to_period(period)).agg(
        # Number of new projects in a given time period
        no_of_projects=("project_id", "count"),
        # Total amount of research funding in a given time period
        amount_total=("amount", "sum"),
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    return impute_empty_periods(
        grouped.reset_index().assign(amount_total=lambda x: x.amount_total / 1000),
        "time_period",
        period,
        min_year,
        max_year,
    )


def gtr_funding_median_per_period(
    gtr_docs: pd.DataFrame, period: str, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Given a table with projects and their funding, return median funding by period

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        period: Time period to group the data by, 'month', 'quarter' or 'year'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'period' - time period
            'amount_median' - median amount of research funding in a given period
    """
    # Check and reformat period
    check_valid(period, ["year", "month", "quarter"])
    period = period[0].capitalize()
    # Convert project start dates to time period
    gtr_docs = (
        gtr_docs.copy()
        .astype({"start": "datetime64[ns]"})
        .assign(time_period=lambda x: x.start.dt.strftime("%Y-%m-%d"))
        .astype({"time_period": "datetime64[ns]"})
    )

    # Group by time period
    grouped = gtr_docs.groupby(gtr_docs["time_period"].dt.to_period(period)).agg(
        # Median project funding in a given period
        amount_median=("amount", np.median),
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    return impute_empty_periods(
        grouped.reset_index().assign(
            amount_median=lambda x: x.amount_median / 1000,
        ),
        "time_period",
        period,
        min_year,
        max_year,
    )


def gtr_get_all_timeseries_period(
    gtr_docs: pd.DataFrame, period: str, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Calculates all typical time series from a list of GtR projects and return
    as one combined table

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        period: Time period to group the data by, 'month', 'quarter' or 'year'

    Returns:
        Dataframe with columns for 'time_period', 'no_of_projects', 'amount_total' and
        'amount_median'
    """
    # Check and reformat period
    check_valid(period, ["year", "month", "quarter"])
    period = period[0].capitalize()
    # Deduplicate projects. This is used to report the number of new projects
    # started each period, accounting for cases where the same project has received
    # additional funding in later periods
    gtr_docs_dedup = gtr_deduplicate_projects(gtr_docs)
    # Number of new projects per time period
    time_series_projects = gtr_funding_per_period(
        gtr_docs_dedup, period, min_year, max_year
    )[["time_period", "no_of_projects"]]
    # Amount of research funding per period (note: here we use the non-duplicated table,
    # to account for additional funding for projects that might have started in earlier periods
    time_series_funding = gtr_funding_per_period(gtr_docs, period, min_year, max_year)
    # Join up both tables
    time_series_funding["no_of_projects"] = time_series_projects["no_of_projects"]
    return time_series_funding


### Crunchbase specific utils


def cb_orgs_founded_per_period(
    cb_orgs: pd.DataFrame, period: str, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Calculates the number of Crunchbase organisations founded per period

    Args:
        cb_orgs: A dataframe with columns for 'id' and 'founded_on' among other data
        period: Time period the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'time_period',
            'no_of_orgs_founded' - number of new organisations founded in a given time period
    """
    # Remove orgs that don't have year when they were founded
    cb_orgs = (
        cb_orgs[-cb_orgs.founded_on.isnull()]
        .copy()
        .assign(time_period=lambda x: pd.to_datetime(x.founded_on))
    )
    # Group by time period
    grouped = cb_orgs.groupby(cb_orgs["time_period"].dt.to_period(period)).agg(
        no_of_orgs_founded=("id", "count")
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    return impute_empty_periods(
        grouped.reset_index(), "time_period", period, min_year, max_year
    )


def cb_investments_per_period(
    cb_funding_rounds: pd.DataFrame, period: str, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Aggregates the raised investment amount and number of deals across all orgs

    Args:
        cb_funding_rounds: A dataframe with columns for 'funding_round_id', 'raised_amount_usd'
            'raised_amount_gbp' and 'announced_on' among other data
        period: Time period the data is grouped by, 'M', 'Q' or 'Y'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'time_period',
            'no_of_rounds' - number of funding rounds (deals) in a given year
            'raised_amount_usd_total' - total raised investment (USD) in a given year
            'raised_amount_gbp_total' - total raised investment (GBP) in a given year
    """
    # Create time period column
    cb_funding_rounds["time_period"] = pd.to_datetime(cb_funding_rounds.announced_on)
    # Group by time period
    grouped = cb_funding_rounds.groupby(
        cb_funding_rounds["time_period"].dt.to_period(period)
    ).agg(
        no_of_rounds=("funding_round_id", "count"),
        raised_amount_usd_total=("raised_amount_usd", "sum"),
        raised_amount_gbp_total=("raised_amount_gbp", "sum"),
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    return impute_empty_periods(
        grouped.reset_index(), "time_period", period, min_year, max_year
    )


def cb_get_all_timeseries(
    cb_orgs: pd.DataFrame,
    cb_funding_rounds: pd.DataFrame,
    period: str,
    min_year: int,
    max_year: int,
) -> pd.DataFrame:
    """
    Combines crunchbase organisations and deals data to produce time series data
    for funding rounds, orgs foundded and amount raised

    Args:
        cb_orgs: A dataframe with columns for 'id' and 'founded_on' among other data
        cb_funding_rounds: A dataframe with columns for 'funding_round_id', 'raised_amount_usd'
            'raised_amount_gbp' and 'announced_on' among other data
        period: Time period to group the data by, 'month', 'quarter' or 'year'
        min_year: Earliest year to impute values for
        max_year: Last year to impute values for

    Returns:
        A dataframe with the following columns:
            'time_period',
            'no_of_rounds' - number of funding rounds (deals) in a given year
            'raised_amount_usd_total' - total raised investment (USD) in a given year
            'raised_amount_gbp_total' - total raised investment (GBP) in a given year
            'no_of_orgs_founded' - number of new organisations founded in a given year
    """
    # Check and reformat period
    check_valid(period, ["year", "month", "quarter"])
    period = period[0].capitalize()
    # Number of new companies per year
    time_series_orgs_founded = cb_orgs_founded_per_period(
        cb_orgs, period, min_year, max_year
    )
    # Amount of raised investment per year
    time_series_investment = cb_investments_per_period(
        cb_funding_rounds, period, min_year, max_year
    )
    # Join up both tables
    time_series_investment["no_of_orgs_founded"] = time_series_orgs_founded[
        "no_of_orgs_founded"
    ]
    return time_series_investment


def sort_companies_by_funding(
    cb_orgs: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Args:
        cb_orgs: A dataframe with a column for 'total_funding_usd' among other data
        verbose: If True, will print what percentage of organisations that have funding info

    Returns:
        A dataframe with organisations sorted by the total funding amount
    """
    df = (
        cb_orgs
        # Add zeros for companies without funding info
        .fillna({"total_funding_usd": 0})
        # Covert all funding values to float
        .assign(
            total_funding_usd=lambda x: x.total_funding_usd.astype(float)
        ).sort_values("total_funding_usd", ascending=False)
    )
    if verbose:
        percent_with_funding = (
            len(cb_orgs.dropna(subset=["total_funding_usd"])) / len(cb_orgs) * 100
        )
        logging.info(
            f"{percent_with_funding:.0f}% of organisations have funding information."
        )
    #         logging.info('oops')
    return df


### Time series trends


def moving_average(
    timeseries_df: pd.DataFrame, window: int = 3, replace_columns: bool = False
) -> pd.DataFrame:
    """
    Calculates rolling mean of yearly timeseries (not centered)

    Args:
        timeseries_df: Should have a 'year' column and at least one other data column
        window: Window of the rolling mean
        rename_cols: If True, will create new set of columns for the moving average
            values with the name pattern `{column_name}_sma{window}` where sma
            stands for 'simple moving average'; otherwise this will replace the original columns

    Returns:
        Dataframe with moving average values
    """
    # Rolling mean
    df_ma = timeseries_df.rolling(window, min_periods=1).mean().drop("year", axis=1)
    # Create new renamed columns
    if not replace_columns:
        column_names = timeseries_df.drop("year", axis=1).columns
        new_column_names = ["{}_sma{}".format(s, window) for s in column_names]
        df_ma = df_ma.rename(columns=dict(zip(column_names, new_column_names)))
        return pd.concat([timeseries_df, df_ma], axis=1)
    else:
        return pd.concat([timeseries_df[["year"]], df_ma], axis=1)


def magnitude(time_series: pd.DataFrame, year_start: int, year_end: int) -> pd.Series:
    """
    Calculates signals' magnitude (i.e. mean across year_start and year_end)

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window

    Returns:
        Series with magnitude estimates for all data columns
    """
    magnitude = time_series.set_index("year").loc[year_start:year_end, :].mean()
    return magnitude


def percentage_change(initial_value, new_value):
    """Calculates percentage change from first_value to second_value"""
    return (new_value - initial_value) / initial_value * 100


def smoothed_growth(
    time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3
) -> pd.Series:
    """Calculates a growth estimate by using smoothed (rolling mean) time series

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate

    Returns:
        Series with smoothed growth estimates for all data columns
    """
    # Smooth timeseries
    ma_df = moving_average(time_series, window, replace_columns=True).set_index("year")
    # Percentage change
    return percentage_change(
        initial_value=ma_df.loc[year_start, :], new_value=ma_df.loc[year_end, :]
    )


def estimate_magnitude_growth(
    time_series: pd.DataFrame, year_start: int, year_end: int, window: int = 3
) -> pd.DataFrame:
    """
    Calculates signals' magnitude, estimates their growth and returns a combined dataframe

    Args:
        time_series: A dataframe with a columns for 'year' and other data
        year_start: First year of the trend window
        year_end: Last year of the trend window
        window: Moving average windows size (in years) for the smoothed growth estimate

    Returns:
        Dataframe with magnitude and growth trend estimates; magnitude is in
        absolute units (e.g. GBP 1000s if analysing research funding) whereas
        growth is expresed as a percentage
    """
    magnitude_df = magnitude(time_series, year_start, year_end)
    growth_df = smoothed_growth(time_series, year_start, year_end, window)
    combined_df = (
        pd.DataFrame([magnitude_df, growth_df], index=["magnitude", "growth"])
        .reset_index()
        .rename(columns={"index": "trend"})
    )
    return combined_df


def filter_years(
    df: pd.DataFrame, date_col: str, keep_year_col: bool, min_year: int, max_year: int
) -> pd.DataFrame:
    """
    Filter dataframe to contain rows within a range of years

    Args:
        df: Dataframe to filter
        date_col: Column containing dates to filter on
        keep_year_col: Keep year column that is generated
        min_year: Lower bound for years to keep
        max_year: Upper bound for years to keep

    Returns:
        Dataframe containing only rows within min_year and max_year bounds
    """
    filtered_df = df.assign(year=lambda x: pd.to_datetime(x[date_col]).dt.year).query(
        f"{min_year}<=year<={max_year}"
    )
    return filtered_df if keep_year_col else filtered_df.drop(columns=["year"])
