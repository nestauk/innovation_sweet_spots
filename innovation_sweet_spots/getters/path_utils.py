"""
innovation_sweet_spots.getters.path_utils

Facilitates easy access to the important input folder paths

"""

from innovation_sweet_spots import PROJECT_DIR

# Input paths
INPUTS_PATH = PROJECT_DIR / "inputs/data/"
GTR_PATH = INPUTS_PATH / "gtr"
CB_PATH = INPUTS_PATH / "cb"
HANSARD_PATH = INPUTS_PATH / "hansard"
GUARDIAN_PATH = INPUTS_PATH / "guardian"

# Output paths
OUTPUT_DATA_PATH = PROJECT_DIR / "outputs/data/"
OUTPUT_GTR_PATH = OUTPUT_DATA_PATH / "gtr"
OUTPUT_CB_PATH = OUTPUT_DATA_PATH / "cb"
PILOT_OUTPUTS = PROJECT_DIR / "outputs/finals/pilot_outputs/"

# Model paths
PILOT_MODELS_DIR = PILOT_OUTPUTS / "models/"
