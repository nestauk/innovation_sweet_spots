"""
innovation_sweet_spots.utils.geo

Utils for geocoding places.

NB: To use Microsoft Bing API, you need to provide the location of your API key
in the .env file, by adding the following line:
export BING_API_KEY=path/to/key
"""
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_json
import geocoder
import os
import dotenv
import pandas as pd

# BING API key location
dotenv.load_dotenv(PROJECT_DIR / ".env")
API_KEY = open(os.environ["BING_API_KEY"], "r").read()

# Crunchbase location_id mapped to NUTS2 regions
CB_NUTS_PATH = PROJECT_DIR / "inputs/data/misc/geo/cb_geos.json"


def geolocate_address(address: str, key: str = API_KEY) -> dict:
    """
    Uses Bing API to geolocate an address

    Args:
        address: A string with an address
        key: Bing API key

    Returns:
        A dict object with location information
    """
    return geocoder.bing(address, key=API_KEY).json


def get_crunchbase_nuts(file_path=CB_NUTS_PATH) -> dict:
    """
    Loads a dict with Crunchbase locations mapped
    to NUTS2 regions (NUTS versions: 2010, 2013 and 2016)

    Retruns:
        A dict of the format {location_id: {NUTS_version: NUTS_region}}
    """
    return load_json(CB_NUTS_PATH)


def add_nuts_to_crunchbase(
    df: pd.DataFrame, cb_nuts: dict, version: int
) -> pd.DataFrame:
    """
    Adds a NUTS2 region column to Crunchbase organisation table
    Versions can be 2010, 2013, 2016
    """
    df[f"nuts2_{version}"] = df.location_id.apply(
        lambda x: cb_nuts[x][f"nuts2_{version}"] if x in cb_nuts else -1
    )
    return df
