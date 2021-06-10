# Build discipline labelled dataset and classify projects into disciplines

import logging
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer


from innovation_sweet_spots import config, PROJECT_DIR
from innovation_sweet_spots.pipeline.make_research_topic_partition import (
    make_network_analysis_inputs,
)
from innovation_sweet_spots.pipeline.prediction_utils import (
    grid_search,
    make_doc_term_matrix,
    make_predicted_label_df,
    parse_parametres,
)
from innovation_sweet_spots.utils.io import get_lookup

warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_labelled_dataset(
    categories_projects: pd.DataFrame, projects: pd.DataFrame, discipline_names: dict
) -> pd.DataFrame:
    """Create labelled dataset"""
    categories_projects = categories_projects.reset_index(name="categories")
    categories_projects["discipline_list"] = [
        [discipline_names[c] for c in cat] for cat in categories_projects["categories"]
    ]

    categories_projects["top_disc"] = [
        pd.Series(x).value_counts().idxmax()
        for x in categories_projects["discipline_list"]
    ]

    project_discipline_lookup = categories_projects.set_index("project_id")[
        "top_disc"
    ].to_dict()

    # Label medical projects
    med_lookup = {
        row["project_id"]: "medical"
        for _, row in projects.iterrows()
        if row["leadFunder"] == "MRC"
    }

    project_discipline_lookup = dict(**project_discipline_lookup, **med_lookup)

    project_labelled = (
        projects.dropna(axis=0, subset=["abstractText"])
        .assign(abstr_length=lambda df: [len(abst) for abst in df["abstractText"]])
        .query("abstr_length>300")
        .assign(
            single_discipline=lambda df: df["project_id"].map(project_discipline_lookup)
        )
        .dropna(axis=0, subset=["single_discipline"])
        .drop(axis=1, labels=["abstr_length"])
        .reset_index(drop=True)
    )

    return project_labelled


if __name__ == "__main__":

    logging.info("Reading data")
    projects, categories_projects = make_network_analysis_inputs()

    comm_project_lookup = get_lookup("outputs/data/gtr/topic_community_lookup")
    discipline_names = config["discipline_classification"]["discipline_names"]
    comm_discipline_names = {
        k: discipline_names[v] for k, v in comm_project_lookup.items()
    }

    cat_coocc = categories_projects.groupby("project_id")["text"].apply(
        lambda x: list(x)
    )

    logging.info("Creating labelled dataset")
    project_labelled = make_labelled_dataset(cat_coocc, projects, comm_discipline_names)

    # Classification and validation
    logging.info("Starting modelling")
    model_parametres = parse_parametres(
        config["discipline_classification"]["model_parametres"]
    )

    Y = np.array(pd.get_dummies(project_labelled["single_discipline"]))
    y_cols = sorted(set(project_labelled["single_discipline"]))

    vect_fit, X_proc = make_doc_term_matrix(
        project_labelled["abstractText"], CountVectorizer, max_features=20000
    )

    f1_multi = make_scorer(f1_score, average="weighted")

    results = []
    models = [
        LogisticRegression(solver="liblinear"),
        RandomForestClassifier(),
        #    GradientBoostingClassifier(),
    ]
    names = ["logistic", "random_forest"]
    # "gradient_boost"]

    for mod, pars, name in zip(models, model_parametres, names):
        logging.info(f"grid searching {name}")
        clf = grid_search(X_proc, Y, mod, pars, f1_multi)
        results.append(clf)

    scores = [r.best_score_ for r in results]
    index_best = scores.index(max(scores))

    logging.info(f"Best classifier is {names[index_best]}")

    best_estimator = results[index_best].best_estimator_
    logging.info(f"{best_estimator}")

    # Predict for all projects with long text descriptions
    pred_df = make_predicted_label_df(projects, vect_fit, best_estimator, y_cols)

    logging.info("Saving results")
    pred_df.to_csv(f"{PROJECT_DIR}/outputs/data/gtr/predicted_disciplines.csv")
    with open(
        f"{PROJECT_DIR}/outputs/models/gtr_discipline_prediction.p", "wb"
    ) as outfile:
        pickle.dump(best_estimator, outfile)
