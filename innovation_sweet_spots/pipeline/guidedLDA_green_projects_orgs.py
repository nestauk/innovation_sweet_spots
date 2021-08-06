from innovation_sweet_spots import logging, PROJECT_DIR
import pandas as pd
import innovation_sweet_spots.analysis.guided_topics as guided_topics
import innovation_sweet_spots.analysis.topic_analysis as iss_topics
from innovation_sweet_spots.utils.io import save_pickle
import numpy as np

RESULTS_DIR = PROJECT_DIR / "outputs/data/results_july"
RUN = "July2021_projects_orgs_stopwords_e400"
TOP2VEC_TABLE = "Karlis copy of clusters_KK_checked LS22Jul21.xlsx"

DEF_SAVE_PATH = RESULTS_DIR / f"guidedLDA_{RUN}.p"

PARAMETERS = {"n_topics": 50, "n_iter": 200, "random_state": 7, "refresh": 20}
SEED_CONFIDENCE = 0.95


def import_documents():
    # Import top2vec model
    top2vec_model = iss_topics.get_top2vec_model(RUN)

    # Import topic seeds
    topic_seeds_df = pd.read_excel(RESULTS_DIR / "topic_seeds.xlsx")
    topic_seeds = [s.strip().split() for s in topic_seeds_df.topic_words.to_list()]

    ## Import and select documents for training the guided LDA model

    # Documents used for top2vec
    clustering = iss_topics.get_clustering(RUN)
    clusters_kk = pd.read_excel(RESULTS_DIR / TOP2VEC_TABLE)
    # Clusters to be further analysed
    selected_clusters = clusters_kk[
        clusters_kk.manual_category != "Climate Research"
    ].cluster_id_level_3.to_list()
    # Documents to be used for guided LDA
    selected_docs = clustering[clustering.cluster_id.isin(selected_clusters)]
    selected_indexes = list(selected_docs.index)
    # Tokenised documents
    tokenised_documents = [top2vec_model.documents[i] for i in selected_indexes]
    return selected_docs, tokenised_documents, topic_seeds


if __name__ == "__main__":
    # Prepare inputs to the model
    logging.info("Importing and preparing inputs")
    selected_docs, tokenised_documents, topic_seeds = import_documents()
    (
        doc_token_matrix,
        topic_seed_dict,
        vocab,
        vectorizer,
    ) = guided_topics.prepare_model_inputs(tokenised_documents, topic_seeds)
    # Train the model
    guidedLDA_model = guided_topics.run_model(
        doc_token_matrix, topic_seed_dict, PARAMETERS, SEED_CONFIDENCE
    )
    # Generate topic keyword list
    topic_keywords = guided_topics.get_topic_keywords(guidedLDA_model, vocab)
    # Generate a visualisation
    guided_topics.plot_pyLDAvis(
        guidedLDA_model,
        vocab,
        doc_token_matrix,
        fpath=RESULTS_DIR / f"LDAvis_{RUN}.html",
    )
    # Save the model data
    data = {
        "docs": selected_docs,
        "vectorizer": vectorizer,
        "model": guidedLDA_model,
        "topics": topic_keywords,
    }
    save_pickle(data, DEF_SAVE_PATH)
