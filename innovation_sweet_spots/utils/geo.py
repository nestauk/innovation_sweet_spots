"""
innovation_sweet_spots.utils.geo

Utils for geocoding places using the open Microsoft Bing API
NB: You need to provide the location of your API key in the .env file,
by adding the following line:

export BING_API_KEY=path/to/key
"""
from innovation_sweet_spots import PROJECT_DIR
import geocoder
import os
import dotenv
import pandas as pd

dotenv.load_dotenv(PROJECT_DIR / ".env")
API_KEY = open(os.environ["BING_API_KEY"], "r").read()

# National Statistics Postcode Lookup (NSPL) location
NSPL_PATH = (
    PROJECT_DIR / "inputs/data/misc/geo/NSPL_FEB_2021_UK/Data/NSPL_FEB_2021_UK.csv"
)


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


def get_nspl(file_path=NSPL_PATH, nrows: int = None) -> pd.DataFrame:
    """
    Loads the National Statistics Postcode Lookup (NSPL) table
    """
    return pd.read_csv(NSPL_PATH, nrows=nrows)
