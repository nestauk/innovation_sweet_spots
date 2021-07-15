"""
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.analysis.green_document_utils import (
    get_green_keywords,
    find_green_gtr_projects,
    find_green_cb_companies,
)
from innovation_sweet_spots.utils.io import save_pickle
from innovation_sweet_spots.utils.text_pre_processing import pre_process_corpus
from innovation_sweet_spots.analysis.analysis_utils import (
    create_documents_from_dataframe,
)

if __name__ == "__main__":

    logging.info("Green document identification and tokenisation pipeline")

    # Identify green projects and companies
    green_keywords = get_green_keywords(clean=True)
    green_projects = find_green_gtr_projects(green_keywords)
    green_orgs = find_green_cb_companies()

    # Create documents
    green_project_texts = create_documents_from_dataframe(
        green_projects,
        columns=["title", "abstractText", "techAbstractText"],
        preprocessor=(lambda x: x),
    )
    green_org_texts = create_documents_from_dataframe(
        green_orgs,
        columns=["short_description", "long_description"],
        preprocessor=(lambda x: x),
    )

    # Train an ngram phraser on both companies and project descriptions
    green_texts = green_project_texts + green_org_texts
    corpus, ngram_phraser = pre_process_corpus(green_texts)

    # Save tokenised corpus files
    corpus_gtr = dict(
        zip(green_projects.project_id.to_list(), corpus[0 : len(green_project_texts)])
    )
    corpus_cb = dict(zip(green_orgs.id.to_list(), corpus[len(green_project_texts) :]))

    # Save the tokeniser and corpus files
    save_pickle(ngram_phraser, PROJECT_DIR / f"outputs/models/bigram_phraser_gtr_cb.p")
    save_pickle(corpus_gtr, PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p")
    save_pickle(corpus_cb, PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised.p")
