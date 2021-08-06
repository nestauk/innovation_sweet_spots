from innovation_sweet_spots.utils.io import load_pickle, save_pickle
from innovation_sweet_spots import PROJECT_DIR, logging

import innovation_sweet_spots.getters.gtr as gtr
from innovation_sweet_spots.analysis.analysis_utils import (
    create_documents_from_dataframe,
)
from innovation_sweet_spots.utils.text_pre_processing import ngrammer, process_text
from innovation_sweet_spots.analysis.text_analysis import (
    setup_spacy_model,
    DEF_LANGUAGE_MODEL,
)

nlp = setup_spacy_model(DEF_LANGUAGE_MODEL)

from innovation_sweet_spots.analysis.green_document_utils import (
    find_green_gtr_projects,
)

import numpy as np
import pandas as pd

# from tqdm.notebook import tqdm
from tqdm import tqdm

RESULTS_DIR = PROJECT_DIR / "outputs/data/results_july"


def check_project_topic(tokenised_text, return_dataframe=True):
    txt = ngrammer(tokenised_text, ngram_phraser, nlp)
    x = guided_topic_dict["vectorizer"].transform([txt])
    probs = model.transform(x)[0]
    if return_dataframe:
        topic_df = pd.DataFrame(
            data={"topic": guided_topic_dict["topics"], "probability": probs}
        ).sort_values("probability", ascending=False)
        return topic_df
    else:
        return probs


if __name__ == "__main__":
    # Import guided topics
    guided_topic_dict = load_pickle(
        RESULTS_DIR / "guidedLDA_July2021_projects_orgs_stopwords_e400.p"
    )
    model = guided_topic_dict["model"]

    # Import all GTR projects
    gtr_projects = gtr.get_gtr_projects()

    # Green projects
    green_projects = find_green_gtr_projects()

    # BUGGGG
    projects_to_check = gtr_projects[-gtr_projects.project_id.isin(green_projects)]

    # Create documents
    project_to_check_texts = create_documents_from_dataframe(
        projects_to_check,
        columns=["title", "abstractText", "techAbstractText"],
        preprocessor=(lambda x: x),
    )

    # Import text processor
    ngram_phraser = load_pickle(PROJECT_DIR / "outputs/models/bigram_phraser_gtr_cb.p")

    logging.disable()
    docs = project_to_check_texts
    project_probs = []
    for doc in tqdm(docs, total=len(docs)):
        project_probs.append(check_project_topic(doc, return_dataframe=False))
    logging.disable(0)

    probs_dict = {
        "projects": projects_to_check[["project_id", "title"]],
        "probs": np.array(project_probs),
        "topics": guided_topic_dict["topics"],
    }
    save_pickle(probs_dict, RESULTS_DIR / "false_positive_check.p")
