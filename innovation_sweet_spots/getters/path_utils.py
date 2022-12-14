"""
innovation_sweet_spots.getters.path_utils

Facilitates easy access to the important input folder paths

"""

from innovation_sweet_spots import PROJECT_DIR

# Input paths
INPUTS_PATH = PROJECT_DIR / "inputs/data/"
GTR_PATH = INPUTS_PATH / "gtr"
GTR_2022_PATH = INPUTS_PATH / "gtr_2022_august"
CB_PATH = INPUTS_PATH / "cb"
CB_GTR_LINK_PATH = INPUTS_PATH / "cb_gtr_link"
HANSARD_PATH = INPUTS_PATH / "hansard"
GUARDIAN_PATH = INPUTS_PATH / "guardian"
DEALROOM_PATH = INPUTS_PATH / "dealroom"
OPENALEX_PATH = INPUTS_PATH / "openAlex"

# Output paths
OUTPUT_PATH = PROJECT_DIR / "outputs"
OUTPUT_DATA_PATH = OUTPUT_PATH / "data"
OUTPUT_GTR_PATH = OUTPUT_DATA_PATH / "gtr"
OUTPUT_CB_PATH = OUTPUT_DATA_PATH / "cb"
PILOT_OUTPUTS = PROJECT_DIR / "outputs/finals/pilot_outputs/"
AIRTABLE_PATH = OUTPUT_PATH / "airtable"

# Model paths
PILOT_MODELS_DIR = PILOT_OUTPUTS / "models/"
