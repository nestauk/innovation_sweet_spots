"""
innovation_sweet_spots.getters.preprocessed

Module for easy access to preprocessed data

"""
from innovation_sweet_spots.getters.path_utils import PILOT_OUTPUTS, OUTPUT_PATH
from innovation_sweet_spots.utils.io import load_pickle
from typing import Dict, Iterator

PILOT_GTR_CORPUS_PATH = PILOT_OUTPUTS / "preprocessed/gtr_abstracts_tokenised.p"
PILOT_CB_CORPUS_PATH = PILOT_OUTPUTS / "preprocessed/cb_descriptions_uk_tokenised.p"

CRUNCHBASE_FULL_PATH = OUTPUT_PATH / "preprocessed/tokens_cb_descriptions_v2022.p"


def get_pilot_gtr_corpus(
    filepath=PILOT_GTR_CORPUS_PATH,
) -> Dict[str, Iterator[str]]:
    """
    Loads a dictionary of tokenised GtR project abstract text,
    following the format {project_id: list of tokens}
    """
    return load_pickle(filepath)


def get_pilot_crunchbase_corpus(
    filepath=PILOT_CB_CORPUS_PATH,
) -> Dict[str, Iterator[str]]:
    """
    Loads a dictionary of tokenised Crunchbase organisation
    descriptions, following the format {id: list of tokens};
    NB: United Kingdom organisations only
    """
    return load_pickle(filepath)


def get_full_crunchbase_corpus(
    filepath=CRUNCHBASE_FULL_PATH,
) -> Dict[str, Iterator[str]]:
    """
    Loads a dictionary of tokenised Crunchbase organisation
    descriptions, following the format {id: list of tokens};
    """
    return load_pickle(filepath)
