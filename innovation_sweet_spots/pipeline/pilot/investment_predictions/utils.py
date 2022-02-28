import pandas as pd
import numpy as np


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


def dedupe_descriptions(cb_data: pd.DataFrame) -> pd.DataFrame:
    """For rows which are duplicates except for differences in their description
    column, drop the rows with no description or the shorter description"""
    dedupe_descs = (
        cb_data.groupby("id")
        .agg(
            {
                "description": lambda x: max(
                    [str(np.nan_to_num(desc, 0)) for desc in x], key=len
                )
            }
        )
        .reset_index()
    )
    return (
        cb_data.drop(columns=["description"])
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
    """Add columns for acquired on dates from the acqisitions data"""
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
        end_date: Widnow end date
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
    cb_data["n_funding_rounds_in_window"] = (
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


def add_n_month_since_last_investment_in_window(
    cb_data: pd.DataFrame, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add column for number of months since last investment in the time window

    Args:
        cb_data: Dataframe to add number of months since last investment to
        start_date: Start date of the time window
        end_date: End date of the time window

    Returns:
        Dataframe with column added for number of months since last investment
        in the time window
    """
    cols_to_loop = cb_data.columns[cb_data.columns.str.contains("funding_round_date")]
    latest_investment_window_date = (
        cb_data[cols_to_loop][
            pd.DataFrame(
                [
                    pd.to_datetime(cb_data[col]).between(
                        start_date, end_date, inclusive="both"
                    )
                    for col in cols_to_loop
                ]
            ).transpose()
        ]
        .astype("datetime64[ns]")
        .max(axis=1)
    )
    cb_data["n_months_since_last_investment_in_window"] = (
        ((end_date - latest_investment_window_date) / np.timedelta64(1, "M"))
        .fillna(-1)
        .astype(int)
    )
    return cb_data


def add_n_month_since_founded(
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
    cb_data["n_month_since_founded"] = (
        ((end_date - pd.to_datetime(cb_data["founded_on"])) / np.timedelta64(1, "M"))
        .fillna(-1)
        .astype(int)
    )
    return cb_data


def drop_multi_cols(
    cb_data: pd.DataFrame, cols_to_drop_str_containing: list
) -> pd.DataFrame:
    """Drop columns containing strings in list 'cols_to_drop_str_containing' provided"""
    return cb_data.drop(
        columns=cb_data.columns[
            cb_data.columns.str.contains("|".join(cols_to_drop_str_containing))
        ]
    )
