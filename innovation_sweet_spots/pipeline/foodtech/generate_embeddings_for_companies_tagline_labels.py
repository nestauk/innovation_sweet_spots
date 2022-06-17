from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.getters.dealroom as dlr
import numpy as np
import innovation_sweet_spots.utils.embeddings_utils as eu
import innovation_sweet_spots.analysis.wrangling_utils as wu
import innovation_sweet_spots.utils.text_cleaning_utils as tcu

DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "foodtech_may2022_companies_tagline_labels"

WEIGHT_TAGLINE = 0.5

if __name__ == "__main__":
    # Get embeddings
    v_labels = dlr.get_label_embeddings()
    v_companies = dlr.get_company_embeddings()

    # Load company data
    DR = wu.DealroomWrangler(dataset="foodtech")

    # Produce lists of company labels
    company_labels_list = (
        DR.company_labels.assign(
            Category=lambda df: df.Category.apply(tcu.clean_dealroom_labels)
        )
        .groupby("id")["Category"]
        .apply(list)
    )

    # Create one embedding vector based on company labels / categories
    category_vectors = []
    # For each company
    for i in v_companies.vector_ids:
        # If the company has labels, average those embeddings
        if i in company_labels_list.index.to_list():
            category_vectors.append(
                v_labels.select_vectors(company_labels_list.loc[i]).mean(axis=0)
            )
        # If the company doesn't have labels, simply take the tagline embedding
        else:
            category_vectors.append(v_companies.select_vectors([i]).mean(axis=0))
    category_vectors = np.array(category_vectors)

    # Combine company tagline and category vector
    vectors = (WEIGHT_TAGLINE * v_companies.vectors) + (
        (1 - WEIGHT_TAGLINE) * category_vectors
    )

    v_combined = eu.Vectors(
        model_name="all-mpnet-base-v2",
        vectors=vectors,
        vector_ids=v_companies.vector_ids,
        folder=DIR,
        filename=FILENAME,
    )

    v_combined.save_vectors()
