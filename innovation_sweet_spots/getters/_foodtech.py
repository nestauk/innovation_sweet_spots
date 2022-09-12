from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
from pathlib import Path

FOODTECH_FOLDER = PROJECT_DIR / "outputs/foodtech"
GUARDIAN_SEARCHES = FOODTECH_FOLDER / "interim/public_discourse/foodtech_all.csv"


def get_guardian_searches(path: Path = GUARDIAN_SEARCHES):
    """Get results from querying guardian API"""
    return pd.read_csv(path)
