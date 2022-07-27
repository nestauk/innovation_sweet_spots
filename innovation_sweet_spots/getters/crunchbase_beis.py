from innovation_sweet_spots.getters.path_utils import PILOT_OUTPUTS
import pandas as pd
from innovation_sweet_spots.pipeline.pilot.investment_predictions.create_dataset.create_beis_indicators import (
    create_beis_indicators,
)


def get_crunchbase_beis(year) -> pd.DataFrame:
    """Load table with crunchbase org_id, location_id and related beis indicators
    If the file exists locally, will load, otherwise will create

    Args:
        year: Attempt to load BEIS indicators for up to the specified year
            e.g if year is 2018, it will load values for 2017"""
    crunchbase_beis_fp = (
        PILOT_OUTPUTS / f"investment_predictions/company_beis_indicators_{year}.csv"
    )
    if not crunchbase_beis_fp.exists():
        create_beis_indicators(year)
    return pd.read_csv(crunchbase_beis_fp)
