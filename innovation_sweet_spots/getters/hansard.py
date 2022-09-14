from innovation_sweet_spots import PROJECT_DIR
import pandas as pd

HANSARD_PATH = (
    PROJECT_DIR / "inputs/data/hansard/hansard_updates/scrapedxml/HANSARD.csv"
)


def get_debates(nrows: int = None) -> pd.DataFrame:
    """Loads Hansard debates"""
    return pd.read_csv(HANSARD_PATH, dtype=object, nrows=nrows).drop_duplicates(
        "id", keep="first"
    )
