"""
innovation_sweet_spots.getters.processed_foodtech

Module for easy access to processed data related to the food tech project.
This is just a start, and later on could have utils for, eg, loading the final food tech taxonomy file and other outputs.
"""
from innovation_sweet_spots.getters.path_utils import OUTPUT_PATH
import pandas as pd

FOODTECH_PATH = OUTPUT_PATH / "foodtech"
