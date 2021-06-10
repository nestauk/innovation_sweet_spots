# Utilities for discipline and industry prediction
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


def parse_parametres(parametre_list: list) -> list:
    """Parse Nones in the parameter dict"""
    parametre_copy = []

    for el in parametre_list:
        new_dict = {}
        for k, v in el.items():
            new_dict[k] = [par if par != "None" else None for par in v]
        parametre_copy.append(new_dict)

    return parametre_copy


def make_doc_term_matrix(
    training_features, transformer, max_features=50000, rescale=True
):

    # Create and apply tfidf transformer
    vect = transformer(
        ngram_range=[1, 2], stop_words="english", max_features=max_features
    )
    fit = vect.fit(training_features)

    # Create processed text

    X_proc = fit.transform(training_features)

    return fit, X_proc


def grid_search(X, y, model, parametres, metric):

    estimator = OneVsRestClassifier(model)
    clf = GridSearchCV(estimator, parametres, scoring=metric, cv=3)
    clf.fit(X, y)
    return clf


def make_predicted_label_df(
    projects,
    vectoriser,
    estimator,
    y_cols,
    text_var="abstractText",
    id_var="project_id",
    min_length=300,
):

    projects_descr = projects.dropna(axis=0, subset=[text_var])

    projects_long = projects_descr.loc[
        [len(desc) > min_length for desc in projects_descr[text_var]]
    ]

    pl_vect = vectoriser.transform(projects_long[text_var])

    pred = estimator.predict_proba(pl_vect)

    pred_df = pd.DataFrame(pred, index=projects_long[id_var], columns=y_cols)
    return pred_df
