"""
innovation_sweet_spots.getters.models

Module for easy access to trained models

"""
from innovation_sweet_spots.analysis import top2vec_utils
from innovation_sweet_spots.analysis.lda_modelling_utils import load_lda_model_data
from innovation_sweet_spots.getters.path_utils import PILOT_MODELS_DIR

PILOT_TOP2VEC_MODEL = "top2vec_gtr_cb"


def get_pilot_lda_model(model_name: str):
    """
    Loads model and data associated with pilot project's topic models

    Args:
        model_name: Either 'broad_corpus' or 'narrow_corpus'
    """
    return load_lda_model_data(model_name, folder=PILOT_MODELS_DIR / "lda")


def get_pilot_top2vec_model():
    """Loads in pilot project's top2vec model and data associated with it"""
    return top2vec_utils.load_top2vec_model_data(
        model_name=PILOT_TOP2VEC_MODEL, folder=PILOT_MODELS_DIR / "top2vec"
    )


def get_pilot_top2vec_document_clusters():
    """Loads in pilot project's top2vec document clusters"""
    return top2vec_utils.load_document_cluster_data(
        model_name=PILOT_TOP2VEC_MODEL, folder=PILOT_MODELS_DIR / "top2vec"
    )
