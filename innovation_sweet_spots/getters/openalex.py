"""
innovation_sweet_spots.getters.openalex

Module for access to data related to OpenAlex
"""
from innovation_sweet_spots.getters.path_utils import OPENALEX_PATH, PROJECT_DIR
import innovation_sweet_spots.utils.embeddings_utils as eu
import pandas as pd


def get_openalex_concept_list() -> pd.DataFrame:
    """
    Fetches a table of all OpenAlex concepts
    Find more information about concepts here: https://docs.openalex.org/about-the-data/concept
    """
    return pd.read_csv(
        OPENALEX_PATH
        / "concepts/OpenAlex concepts in use (16 June 2022) - concepts.csv"
    )


# Preprocessed embeddings
MODEL = "all-mpnet-base-v2"
DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"


def get_openalex_concept_embeddings(
    model=MODEL, folder=DIR, filename="openAlex_concepts_june2022"
):
    return eu.Vectors(
        model_name=model,
        folder=folder,
        filename=filename,
    )
