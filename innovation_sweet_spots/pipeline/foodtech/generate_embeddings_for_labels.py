"""
Script to generate embedding representations of Dealroom companies

Usage (from the repo root directory):
python innovation_sweet_spots/pipeline/foodtech/generate_embeddings_v1.py
"""
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.analysis.wrangling_utils as wu
import innovation_sweet_spots.utils.embeddings_utils as eu
import innovation_sweet_spots.utils.text_cleaning_utils as tcu
import re

MODEL = "all-mpnet-base-v2"
DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "foodtech_may2022_labels"

if __name__ == "__main__":

    v_labels = eu.Vectors(
        model_name=MODEL,
        folder=DIR,
        filename=FILENAME,
    )

    DR = wu.DealroomWrangler(dataset="foodtech")
    labels_unique = (
        DR.labels.Category.apply(tcu.clean_dealroom_labels).drop_duplicates(
            keep="first"
        )
    ).to_list()

    v_labels.generate_new_vectors(
        new_document_ids=labels_unique,
        texts=labels_unique,
    )

    v_labels.save_vectors()
