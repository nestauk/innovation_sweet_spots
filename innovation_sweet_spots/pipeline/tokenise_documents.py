"""
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import save_pickle
from innovation_sweet_spots.utils.text_pre_processing import pre_process_corpus
from innovation_sweet_spots.analysis.analysis_utils import (
    create_documents_from_dataframe,
)
from innovation_sweet_spots.getters import crunchbase, gtr

if __name__ == "__main__":

    logging.info("Document tokenisation pipeline")

    # Get GTR projects
    gtr_projects = gtr.get_gtr_projects()
    # Get Crunchbase UK orgs
    cb_orgs = crunchbase.get_crunchbase_orgs()

    # Create documents
    gtr_project_texts = create_documents_from_dataframe(
        gtr_projects,
        columns=["title", "abstractText", "techAbstractText"],
        preprocessor=(lambda x: x),
    )

    cb_org_texts = create_documents_from_dataframe(
        cb_orgs,
        columns=["short_description", "long_description"],
        preprocessor=(lambda x: x),
    )

    # Train an ngram phraser on both companies and project descriptions
    texts = gtr_project_texts + cb_org_texts
    logging.info(f"Tokenising {len(texts)} documents")
    corpus, ngram_phraser = pre_process_corpus(
        texts, n_gram=4, min_count=5, threshold=0.35
    )

    # Save tokenised corpus files
    corpus_gtr = dict(
        zip(gtr_projects.project_id.to_list(), corpus[0 : len(gtr_project_texts)])
    )
    corpus_cb = dict(zip(cb_orgs.id.to_list(), corpus[len(gtr_project_texts) :]))

    # Save the tokeniser and corpus files
    save_pickle(
        ngram_phraser, PROJECT_DIR / f"outputs/models/ngram_phraser_gtr_cb_full.p"
    )
    save_pickle(corpus_gtr, PROJECT_DIR / "outputs/data/gtr/gtr_docs_tokenised_full.p")
    save_pickle(corpus_cb, PROJECT_DIR / "outputs/data/cb/cb_docs_tokenised_full.p")
