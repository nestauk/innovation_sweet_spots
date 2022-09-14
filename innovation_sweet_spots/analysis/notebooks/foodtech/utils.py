"""
"""
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd

# EXCLUDED_DEAL_TYPES = [
#     "GRANT",
#     '-',
#     # np.nan,
#     'ICO',
# ]

EARLY_DEAL_TYPES = [
    "SERIES B",
    "EARLY VC",
    "SERIES A",
    "SEED",
    "ANGEL",
    "CONVERTIBLE",
    "LATE VC",
    "SPINOUT",
    "GROWTH EQUITY VC",
    "SERIES C",
    "SERIES D",
    "PRIVATE PLACEMENT VC",
    "SERIES F",
    "SERIES E",
    "SECONDARY",
    "SERIES H",
    "SERIES G",
    "SERIES I",
    "SPAC PRIVATE PLACEMENT",
    "GROWTH EQUITY NON VC",
    "MEDIA FOR EQUITY",
    "PROJECT, REAL ESTATE, INFRASTRUCTURE FINANCE",
]

LATE_DEAL_TYPES = [
    "ACQUISITION",
    "DEBT",
    "POST IPO EQUITY",
    "IPO",
    "MERGER",
    "SPAC IPO",
    "BUYOUT",
    "POST IPO DEBT",
    "POST IPO CONVERTIBLE",
    "LENDING CAPITAL",
    "POST IPO SECONDARY",
]


def result_dict_to_dataframe(
    result_dict: dict, sort_by: str = "counts", category_name: str = "cluster"
) -> pd.DataFrame:
    """Prepares the output dataframe"""
    return (
        pd.DataFrame(result_dict)
        .T.reset_index()
        .sort_values(sort_by)
        .rename(columns={"index": category_name})
    )


def get_category_time_series(
    time_series_df: pd.DataFrame,
    category_of_interest: str,
    time_column: str = "releaseYear",
    category_column: str = "cluster",
) -> dict:
    """Gets cluster or user-specific time series"""
    return (
        time_series_df.query(f"{category_column} == @category_of_interest")
        .drop(category_column, axis=1)
        .sort_values(time_column)
        .rename(columns={time_column: "year"})
        .assign(year=lambda x: pd.to_datetime(x.year.apply(lambda y: str(int(y)))))
        .pipe(
            au.impute_empty_periods,
            time_period_col="year",
            period="Y",
            min_year=2010,
            max_year=2021,
        )
        .assign(year=lambda x: x.year.dt.year)
    )


def get_estimates(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
    estimate_function=au.growth,
    year_start: int = 2019,
    year_end: int = 2020,
):
    """
    Get growth estimate for each category

    growth_estimate_function - either growth, smoothed_growth, or magnitude
    For growth, use 2019 and 2020 as year_start and year_end
    For smoothed_growth and magnitude, use 2017 and 2021
    """
    time_series_df_ = time_series_df[[time_column, category_column, value_column]]

    result_dict = {
        category: estimate_function(
            get_category_time_series(
                time_series_df_, category, time_column, category_column
            ),
            year_start=year_start,
            year_end=year_end,
        )
        for category in time_series_df[category_column].unique()
    }
    return result_dict_to_dataframe(result_dict, value_column, category_column)


def get_magnitude_vs_growth(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
    year_start=2017,
    year_end=2021,
):
    """Get magnitude vs growth esitmates"""
    df_growth = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=au.smoothed_growth,
        year_start=year_start,
        year_end=year_end,
    ).rename(columns={value_column: "Growth"})

    df_magnitude = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=au.magnitude,
        year_start=year_start,
        year_end=year_end,
    ).rename(columns={value_column: "Magnitude"})

    return df_growth.merge(df_magnitude, on=category_column)


def deal_amount_to_range(
    amount: float, currency: str = "£", categories: bool = True
) -> str:
    """
    Convert amounts to range in millions
    Args:
        amount: Investment amount (in GBP thousands)
        categories: If True, adding indicative deal categories
        currency: Currency symbol
    """
    amount /= 1e3
    if (amount >= 0.001) and (amount < 1):
        return f"{currency}0-1M" if not categories else f"{currency}0-1M"
    elif (amount >= 1) and (amount < 4):
        return f"{currency}1-4M" if not categories else f"{currency}1-4M"
    elif (amount >= 4) and (amount < 15):
        return f"{currency}4-15M" if not categories else f"{currency}4-15M"
    elif (amount >= 15) and (amount < 40):
        return f"{currency}15-40M" if not categories else f"{currency}15-40M"
    elif (amount >= 40) and (amount < 100):
        return f"{currency}40-100M" if not categories else f"{currency}40-100M"
    elif (amount >= 100) and (amount < 250):
        return f"{currency}100-250M"
    elif amount >= 250:
        return f"{currency}250+"
    else:
        return "n/a"
