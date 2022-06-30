"""
innovation_sweet_spots.getters.airtable

Module for easy access to Airtable tables
Find more info on using Airtable API here: https://pyairtable.readthedocs.io/en/latest/api.html
"""
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters.path_utils import AIRTABLE_PATH
import os
import dotenv
from functools import lru_cache
from pyairtable import Table

dotenv.load_dotenv(PROJECT_DIR / ".env")
API_KEY = open(os.environ["AIRTABLE_API_KEY"], "r").read()


def get_greentech_table():
    """Fetches the UK green tech company table"""
    return Table(API_KEY, "appwuLQSM6sC4G8fm", "Decarbonising homes (UK)")


def get_test_table():
    """Fetches a test table (for development purposes)"""
    return Table(API_KEY, "appq5pAQoLYA2JMVA", "Table 1")
