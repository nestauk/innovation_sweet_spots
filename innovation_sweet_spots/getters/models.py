"""
innovation_sweet_spots.getters.models

Module for easy access to trained models

"""
from innovation_sweet_spots.analysis.lda_modelling_utils import load_lda_model_data
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.getters.path_utils import PILOT_MODELS_DIR


def get_pilot_lda_model(model_name: str):
    """
    Loads model and data associated with pilot project's topic models

    Args:
        model_name: Either 'broad_corpus' or 'narrow_corpus'
    """
    return load_lda_model_data(model_name, folder=PILOT_MODELS_DIR / "lda")
