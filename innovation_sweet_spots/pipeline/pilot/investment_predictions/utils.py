import pandas as pd
import numpy as np


def convert_col_to_has_col(df: pd.DataFrame, col: str, drop: bool) -> pd.DataFrame:
    df[f"has_{col}"] = df[col].notnull().astype("int")
    if drop:
        return df.drop(columns=[col])
    return df


def tech_cats_to_dummies(cb_data: pd.DataFrame) -> pd.DataFrame:
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


def dedupe_descriptions(cb_data):
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


def add_unstack_data(left, right, left_id, right_id, right_col_to_add, prefix):
    """General function to add a column to the left dataframe
    from the right dataframe. If there are multiple matches in the
    right dataframe, additional columns will be created in the left dataframe
    for each match (numbered in chronological order)

    Example:

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


def add_acquired_on(cb_data, acquired_on_data):
    return add_unstack_data(
        left=cb_data,
        right=acquired_on_data,
        left_id="id",
        right_id="acquiree_id",
        right_col_to_add="acquired_on",
        prefix="acquired_on_",
    )


def add_went_public_on(cb_data, ipo_data):
    return add_unstack_data(
        left=cb_data,
        right=ipo_data,
        left_id="id",
        right_id="org_id",
        right_col_to_add="went_public_on",
        prefix="went_public_on_",
    )


def add_funding_round_ids(cb_data, cb_funding_data):
    cb_fd = cb_funding_data[["org_id", "id"]].rename(columns={"id": "funding_round_id"})
    return add_unstack_data(
        left=cb_data,
        right=cb_fd,
        left_id="id",
        right_id="org_id",
        right_col_to_add="funding_round_id",
        prefix="funding_round_id_",
    )


def add_funding_round_dates(cb_data, cb_funding_data):
    return add_unstack_data(
        left=cb_data,
        right=cb_funding_data,
        left_id="id",
        right_id="org_id",
        right_col_to_add="announced_on",
        prefix="funding_round_date_",
    )


def window_flag(cb_data, start_date, end_date, variable):
    """General window flag"""
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


def future_flag(cb_data, start_date, variable):
    """General future flag"""
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


def add_n_funding_rounds_in_window(cb_data, start_date, end_date):
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


def add_n_month_since_last_investment_in_window(cb_data, start_date, end_date):
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


def add_n_month_since_founded(cb_data, end_date):
    cb_data["n_month_since_founded"] = (
        ((end_date - pd.to_datetime(cb_data["founded_on"])) / np.timedelta64(1, "M"))
        .fillna(-1)
        .astype(int)
    )
    return cb_data


def drop_multi_cols(cb_data, cols_to_drop_str_containing):
    return cb_data.drop(
        columns=cb_data.columns[
            cb_data.columns.str.contains("|".join(cols_to_drop_str_containing))
        ]
    )
