"""
innovation_sweet_spots.getters.beis_indicators

Regional UK research and development indicators. Download the data from:
https://access-research-development-spatial-data.beis.gov.uk/indicators
"""
from innovation_sweet_spots.getters.path_utils import INPUTS_PATH
from innovation_sweet_spots.utils.io import unzip_files
import pandas as pd
import urllib

# Data version to use
VERSION = "0_1_4"
# Location of the data
URL = f"https://access-research-development-spatial-data.beis.gov.uk/data/beis_indicators_{VERSION}.zip"
# Local path to the data
LOCAL_PATH = INPUTS_PATH / f"misc/beis_indicators/beis_indicators_{VERSION}"


def download_beis_indicators():
    """Downloads data from BEIS website"""
    filename = URL.split("/")[-1]
    urllib.request.urlretrieve(URL, LOCAL_PATH.parent / filename)
    unzip_files(LOCAL_PATH.parent / filename, LOCAL_PATH, delete=True)


def get_beis_indicators() -> pd.DataFrame:
    """Loads regional UK research and development indicators"""
    if not LOCAL_PATH.is_dir():
        LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        download_beis_indicators()
    return pd.read_csv(LOCAL_PATH / f"beis_indicators_{VERSION}_NUTS2.csv")
