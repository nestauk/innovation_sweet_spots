"""
Script to generate embedding representations of OpenAlex concepts

Usage (from the repo root directory):
python innovation_sweet_spots/pipeline/foodtech/generate_embeddings_OpenAlex_concepts.py
"""
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.utils.embeddings_utils as eu
from innovation_sweet_spots.getters.openalex import get_openalex_concept_list

MODEL = "all-mpnet-base-v2"
DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
# The embeddings are for the concept version of June 16, 2022
FILENAME = "openAlex_concepts_june2022"

if __name__ == "__main__":

    v_labels = eu.Vectors(
        model_name=MODEL,
        folder=DIR,
        filename=FILENAME,
    )

    concepts = get_openalex_concept_list()

    v_labels.generate_new_vectors(
        new_document_ids=concepts.openalex_id.to_list(),
        texts=concepts.normalized_name.to_list(),
    )

    v_labels.save_vectors()
