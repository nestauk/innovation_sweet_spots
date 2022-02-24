"""
Script to combine Dealroom tables, which were exported from the
database for the food tech horizon scan analysis

Usage (from the repo root directory):
python innovation_sweet_spots/pipeline/foodtech/combine_dealroom_tables.py
"""
from innovation_sweet_spots.getters.path_utils import DEALROOM_PATH
from innovation_sweet_spots.utils.io import combine_tables_from_folder

# Location of the tables exported from Dealroom
INPUTS_PATH = DEALROOM_PATH / "raw_exports/foodtech"
# Location where to save the combined table
OUTPUT_FILEPATH = DEALROOM_PATH / "dealroom_foodtech.csv"

if __name__ == "__main__":

    combine_tables_from_folder(folder=INPUTS_PATH, deduplicate_by=["id"]).to_csv(
        OUTPUT_FILEPATH, index=False
    )
