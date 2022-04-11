import pandas as pd


def merge_matches_with_cb_and_gtr(
    matches: pd.DataFrame, name: str, cb: pd.DataFrame, gtr: pd.DataFrame
) -> pd.DataFrame:
    """Merge fuzzy matches with crunchbase org ids and gateway to research
    ids

    Args:
        matches: Dataframe containing columns x and y relating to the indexes in
            cb and gtr dataframes.
        name: Crunchbase column to rename, "name" or "legal_name"
        cb: Dataframe containing crunchbase organisation ids
        gtr: Dataframe containing gateway to research organisation ids
    """
    name_col_to_drop = "name" if name is "legal_name" else "legal_name"
    return (
        matches.merge(right=cb, right_index=True, left_on="x", how="left")
        .rename(columns={name: "cb_name", "address": "cb_address", "id": "cb_org_id"})
        .drop(columns=["country_code", name_col_to_drop])
        .merge(right=gtr, right_index=True, left_on="y", how="left")
        .rename(
            columns={
                "name": "gtr_name",
                "addresses": "gtr_addresses",
                "id": "gtr_org_id",
            }
        )
        .sort_values(by=["sim_mean"])
        .reset_index(drop=True)
    )
