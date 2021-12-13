"""
Utils for doing data analysis

"""
import pandas as pd
import numpy as np


def impute_empty_years(
    yearly_stats: pd.DataFrame, min_year: int = None, max_year: int = None
) -> pd.DataFrame:
    """
    Imputes zero values for years without data

    Args:
        yearly_stats: A dataframe with a 'year' column and other columns with data
        min_year: Lower bound for years to keep; can be smaller than yearly_stats.year.min()
        max_year: Higher bound for years; can be larger than yearly_stats.year.min()

    Returns:
        A data frame with imputed 0s for years with no data
    """
    min_year, max_year = set_def_min_max_years(yearly_stats, min_year, max_year)
    return (
        pd.DataFrame(data={"year": range(min_year, max_year + 1)})
        .merge(yearly_stats, how="left")
        .fillna(0)
        .astype(yearly_stats.dtypes)
    )


def set_def_min_max_years(df: pd.DataFrame, min_year: int, max_year: int) -> (int, int):
    """Set the default values for min and max years"""
    if min_year is None:
        min_year = df.year.min()
    if max_year is None:
        max_year = df.year.max()
    return min_year, max_year


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


def gtr_funding_per_year(
    gtr_docs: pd.DataFrame, min_year: int = None, max_year: int = None
) -> pd.DataFrame:
    """
    Given a table with projects and their funding, return an aggregation by year

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        min_year: Lower bound for years to keep
        max_year: Higher bound for years to keep

    Returns:
        A dataframe with the following columns:
            'year',
            'no_of_projects' - number of new projects in a given year,
            'amount_total' - total amount of research funding in a given year,
            'amount_median' - median project funding in a given year

    """
    # Convert project start dates to years
    gtr_docs = gtr_docs.copy()
    gtr_docs["year"] = pd.to_datetime(gtr_docs.start).dt.year
    # Set min and max years for aggregation
    min_year, max_year = set_def_min_max_years(gtr_docs, min_year, max_year)
    # Group by year
    return (
        gtr_docs.groupby("year")
        .agg(
            # Number of new projects in a given year
            no_of_projects=("project_id", "count"),
            # Total amount of research funding in a given year
            amount_total=("amount", "sum"),
            # Median project funding in a given year
            amount_median=("amount", np.median),
        )
        .reset_index()
        # Limit results between min and max years
        .query(f"year>={min_year}")
        .query(f"year<={max_year}")
        # Convert to thousands
        .assign(
            amount_total=lambda x: x.amount_total / 1000,
            amount_median=lambda x: x.amount_median / 1000,
        )
        # Add zero values for years without data
        .pipe(impute_empty_years, min_year=min_year, max_year=max_year)
    )


def gtr_get_all_timeseries(
    gtr_docs: pd.DataFrame, min_year: int = None, max_year: int = None
) -> pd.DataFrame:
    """
    Calculates all typical time series from a list of GtR projects and return
    as one combined table

    Args:
        gtr_docs: A dataframe with columns for 'start', 'project_id' and 'amount'
            (research funding) among other project data
        min_year: Lower bound for years to keep
        max_year: Higher bound for years to keep

    Returns:
        Dataframe with columns for 'year', 'no_of_projects', 'amount_total' and
        'amount_median'
    """
    # Deduplicate projects. This is used to report the number of new projects
    # started each year, accounting for cases where the same project has received
    # additional funding in later years
    gtr_docs_dedup = gtr_deduplicate_projects(gtr_docs)
    # Number of new projects per year
    time_series_projects = gtr_funding_per_year(gtr_docs_dedup, min_year, max_year)[
        ["year", "no_of_projects"]
    ]
    # Amount of research funding per year (note: here we use the non-duplicated table,
    # to account for additional funding for projects that might have started in earlier years
    time_series_funding = gtr_funding_per_year(gtr_docs, min_year, max_year)
    # Join up both tables
    time_series_funding["no_of_projects"] = time_series_projects["no_of_projects"]
    return time_series_funding


### Crunchbase specific utils


def cb_orgs_founded_per_year(
    cb_orgs: pd.DataFrame, min_year: int = None, max_year: int = None
) -> pd.DataFrame:
    """
    Calculates the number of Crunchbase organisations founded in a given year

    Args:
        cb_orgs: A dataframe with columns for 'id' and 'founded_on' among other data
        min_year: Lower bound for years to keep
        max_year: Higher bound for years to keep

    Returns:
        A dataframe with the following columns:
            'year',
            'no_of_orgs_founded' - number of new organisations founded in a given year
    """
    # Remove orgs that don't have year when they were founded
    cb_orgs = cb_orgs[-cb_orgs.founded_on.isnull()].copy()
    # Convert dates to years
    cb_orgs["year"] = pd.to_datetime(cb_orgs.founded_on).dt.year
    # Set min and max years for aggregation
    min_year, max_year = set_def_min_max_years(cb_orgs, min_year, max_year)
    # Group by year
    return (
        cb_orgs.groupby("year")
        .agg(no_of_orgs_founded=("id", "count"))
        .reset_index()
        .pipe(impute_empty_years, min_year=min_year, max_year=max_year)
    )


def cb_investments_per_year(
    cb_funding_rounds: pd.DataFrame, min_year: int = None, max_year: int = None
) -> pd.DataFrame:
    """
    Aggregates the raised investment amount and number of deals across all orgs

    Args:
        cb_funding_rounds: A dataframe with columns for 'funding_round_id', 'raised_amount_usd'
            'raised_amount_gbp' and 'announced_on' among other data
        min_year: Lower bound for years to keep
        max_year: Higher bound for years to keep

    Returns:
        A dataframe with the following columns:
            'year',
            'no_of_rounds' - number of funding rounds (deals) in a given year
            'raised_amount_usd_total' - total raised investment (USD) in a given year
            'raised_amount_gbp_total' - total raised investment (GBP) in a given year
    """
    # Convert dates to years
    cb_funding_rounds["year"] = pd.to_datetime(cb_funding_rounds.announced_on).dt.year
    # Set min and max years for aggregation
    min_year, max_year = set_def_min_max_years(cb_funding_rounds, min_year, max_year)
    # Group by year
    return (
        cb_funding_rounds.groupby("year")
        .agg(
            no_of_rounds=("funding_round_id", "count"),
            raised_amount_usd_total=("raised_amount_usd", "sum"),
            raised_amount_gbp_total=("raised_amount_gbp", "sum"),
        )
        .reset_index()
        .query(f"year>={min_year}")
        .query(f"year<={max_year}")
        .pipe(impute_empty_years, min_year=min_year, max_year=max_year)
    )


def cb_get_all_timeseries(
    cb_orgs: pd.DataFrame,
    cb_funding_rounds: pd.DataFrame,
    min_year: int = None,
    max_year: int = None,
) -> pd.DataFrame:
    """
    Calculates all typical time series from a list of GtR projects and return
    as one combined table

    Args:
        cb_orgs: A dataframe with columns for 'id' and 'founded_on' among other data
        cb_funding_rounds: A dataframe with columns for 'funding_round_id', 'raised_amount_usd'
            'raised_amount_gbp' and 'announced_on' among other data
        min_year: Lower bound for years to keep
        max_year: Higher bound for years to keep

    Returns:
        A dataframe with the following columns:
            'year',
            'no_of_rounds' - number of funding rounds (deals) in a given year
            'raised_amount_usd_total' - total raised investment (USD) in a given year
            'raised_amount_gbp_total' - total raised investment (GBP) in a given year
            'no_of_orgs_founded' - number of new organisations founded in a given year
    """
    # Number of new companies per year
    time_series_orgs_founded = cb_orgs_founded_per_year(cb_orgs, min_year, max_year)
    # Amount of raised investment per year
    time_series_investment = cb_investments_per_year(
        cb_funding_rounds, min_year, max_year
    )
    # Join up both tables
    time_series_investment["no_of_orgs_founded"] = time_series_orgs_founded[
        "no_of_orgs_founded"
    ]
    return time_series_investment


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
