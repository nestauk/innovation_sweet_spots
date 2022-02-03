"""
innovation_sweet_spots.getters.preprocessed

Module for easy access to preprocessed data

"""
from innovation_sweet_spots.getters.path_utils import PILOT_OUTPUTS
from innovation_sweet_spots.utils.io import load_pickle
from typing import Dict, Iterator

DEFAULT_GTR_CORPUS_PATH = PILOT_OUTPUTS / "tokenised_data/gtr_docs_tokenised_full.p"


def get_tokenised_gtr_corpus(
    filepath=DEFAULT_GTR_CORPUS_PATH,
) -> Dict[str, Iterator[str]]:
    """GtR a dictionary of tokenised project text, following the format {project_id: list of tokens}"""
    return load_pickle(filepath)
