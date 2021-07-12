import pandas as pd
from innovation_sweet_spots import logging
from innovation_sweet_spots.getters.path_utils import HANSARD_PATH
from typing import Iterable


def get_hansard_data(
    nrows: int = None, n_rows_to_skip: int = 1700000, start_year: int = 2006
) -> pd.DataFrame:
    """
    Loads in the Hansard table.
    The dataset is very large, so you can skip rows to more recent speeches,
    that start around year 2006
    """
    hansard_data = (
        pd.read_csv(
            HANSARD_PATH / "hansard-speeches-v310.csv",
            skiprows=range(1, n_rows_to_skip),
            nrows=nrows,
        )
        .query("speech_class=='Speech'")
        .query(f"year>={start_year}")
    )
    return hansard_data
