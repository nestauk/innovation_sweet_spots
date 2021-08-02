from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots.analysis.green_document_utils import (
    find_green_cb_companies,
    find_green_gtr_projects,
)
import pandas as pd


def get_green_gtr_docs(
    fpath=PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p",
):
    """Get tokenised docs in a dict {id: doc}"""
    return load_pickle(fpath)


def get_green_cb_docs(fpath=PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised.p"):
    """Get tokenised docs in a dict {id: doc}"""
    return load_pickle(fpath)


def get_green_cb_docs_by_country(country="United Kingdom"):
    """Get tokenised docs in a list, and a dataframe"""
    cb_corpus = get_green_cb_docs()
    green_orgs = find_green_cb_companies()
    green_orgs = green_orgs.reset_index(drop=True)
    if country is not None:
        green_country_orgs = green_orgs[green_orgs.country == country]
    else:
        green_country_orgs = green_orgs
    cb_corpus_country = [list(cb_corpus.values())[i] for i in green_country_orgs.index]
    return cb_corpus_country, green_country_orgs


# def get_gtr_cb_green_document_table():
#     # Fetch green projects
#     green_projects = (
#         find_green_gtr_projects()[['project_id', 'title', 'abstractText']]
#         .rename(columns={
#             'project_id': 'doc_id',
#             'abstractText': 'description'})
#         )
#     green_projects['source'] = 'gtr'
#     # Fetch green companies
#     _, green_country_orgs = get_green_cb_docs_by_country(country='United Kingdom')
#     green_country_orgs = (
#         green_country_orgs[['id', 'name', 'short_description']]
#         .rename(columns={
#             'id': 'doc_id',
#             'name': 'title',
#             'short_description': 'description'})
#     )
#     green_country_orgs['source'] = 'cb'
#     doc_table = pd.concat([green_projects, green_country_orgs], axis=1, ignore_index=True)
#     return doc_table
