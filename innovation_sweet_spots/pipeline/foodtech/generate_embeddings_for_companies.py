# %%
"""
Script to generate embedding representations of Dealroom companies

Usage (from the repo root directory):
python innovation_sweet_spots/pipeline/foodtech/generate_embeddings...
"""
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.analysis.wrangling_utils as wu
import innovation_sweet_spots.utils.embeddings_utils as eu
import innovation_sweet_spots.utils.text_cleaning_utils as tcu
import re

# %%
MODEL = "all-mpnet-base-v2"
DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "foodtech_may2022_companies"

# %%
if __name__ == "__main__":

    v_companies = eu.Vectors(
        model_name=MODEL,
        folder=DIR,
        filename=FILENAME,
    )

    DR = wu.DealroomWrangler(dataset="foodtech")
    DR.company_data.TAGLINE = DR.company_data.TAGLINE.fillna("")

    v_companies.generate_new_vectors(
        new_document_ids=DR.company_data.id.to_list(),
        texts=DR.company_data.TAGLINE.to_list(),
    )

    v_companies.save_vectors()
