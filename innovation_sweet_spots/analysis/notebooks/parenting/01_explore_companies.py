# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
import sentence_transformers
from sentence_transformers.SentenceTransformer import SentenceTransformer
from numpy.typing import ArrayLike
from typing import Iterator, Union
from scipy.spatial.distance import cdist
import pandas as pd

from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.analysis.query_categories import query_cb_categories
from innovation_sweet_spots.getters.preprocessed import get_pilot_crunchbase_corpus
from innovation_sweet_spots.utils.io import save_pickle

CB = CrunchbaseWrangler()

# %%
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots.getters.preprocessed import get_full_crunchbase_corpus


# %%
class QueryEmbeddings:
    """
    Helper class to interrogate embeddings and find most similar texts
    to a provided input text

    """

    def __init__(
        self,
        vectors: ArrayLike,
        texts: Iterator[str],
        model: Union[str, SentenceTransformer] = None,
    ):
        """
        Find most similar vectors to the given vector

        Args:
            vectors: Numpy array of shape [n_texts, n_dimensions] encoding texts
            texts: List of text strings (len(texts) must be equal to n_texts)
            model: Either model name or model instance
        """
        self.vectors = vectors
        self.texts = texts
        self.index_to_text = dict(zip(list(range(len(texts))), texts))
        self.text_to_index = dict(zip(texts, list(range(len(texts)))))
        self.model = (
            sentence_transformers.SentenceTransformer(model)
            if type(model) is str
            else model
        )

    def find_most_similar_by_id(self, vector_id: int) -> Iterator[float]:
        """Find most similar vectors to the given vector"""
        similarities = (
            1
            - cdist(self.vectors[vector_id, :].reshape(1, -1), self.vectors, "cosine")[
                0
            ]
        )
        return self.similarities_dataframe(similarities)

    def find_most_similar(self, text: str) -> Iterator[float]:
        """Find most similar vectors to the given text"""
        if text in self.texts:
            return self.find_most_similar_by_id(self.text_to_index[text])
        elif self.model is not None:
            new_vector = self.model.encode([text])
            similarities = (
                1 - cdist(new_vector[0, :].reshape(1, -1), self.vectors, "cosine")[0]
            )
            return self.similarities_dataframe(similarities)
        else:
            raise "Provide embedding model"

    def similarities_dataframe(self, similarities: Iterator[float]) -> pd.DataFrame:
        """Prepare outputs data frame"""
        return pd.DataFrame(
            {
                "text": self.text_to_index.keys(),
                "similarity": similarities,
            }
        ).sort_values("similarity", ascending=False)


# %% [markdown]
# ## Explore relevant CB industries

# %%
embedding_model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

# %%
vectors = embedding_model.encode(CB.industries)

# %%
q = QueryEmbeddings(vectors, CB.industries, embedding_model)

# %%
q.find_most_similar("education").head(30).text.to_list()

# %%
INDUSTRIES = [
    "parenting",
    "fertility",
    "child care",
    "children",
    "underserved children",
    "family",
    "baby",
    "education",
    "toys",
    "edtech",
    "video games",
]

# %% [markdown]
# ## Check CB companies
#
# - Select companies by specific industries, and filter by keywords
# - Additionally, add by keywords

# %%
import innovation_sweet_spots.utils.text_processing_utils as tpu

# %%
query_df = query_cb_categories(INDUSTRIES, CB, return_only_matches=True)

# %%
ids = query_df.id.to_list()
selected_companies = CB.cb_organisations.query(f"id in @ids")

# %%
preprocessed_texts = tpu.create_documents_from_dataframe(
    selected_companies,
    ["name", "short_description", "long_description"],
    preprocessor=tpu.preprocess_clean_text,
)

# %%
corpus = dict(zip(ids, preprocessed_texts))

# %%
save_pickle(
    corpus,
    PROJECT_DIR
    / "outputs/finals/parenting/preprocessed/preprocessed_cb_descriptions_v1.p",
)

# %%
selected_funded_companies = au.sort_companies_by_funding(selected_companies).query(
    "total_funding_usd > 0"
)

# %% [markdown]
# ### Keywords

# %%
corpus_full = get_full_crunchbase_corpus()

# %%
CB.cb_organisations.sample()[["id", "name", "short_description"]]

# %%
INDUSTRIES = [
    "parenting",
    "fertility",
    "child care",
    "children",
    "underserved children",
    "family",
    "baby",
    "education",
    "toys",
    "edtech",
    "video games",
]

# %%
USER_TERMS = [
    ["parent"],
    ["mother"],
    [" mom "],
    ["father"],
    [" dad "],
    ["baby"],
    ["babies"],
    ["infant"],
    ["child"],
    ["toddler"],
    ["preschool"],
    ["pre school"],
    [" kid "],
    ["kids"],
]

LEARN_TERMS = [["learn"], ["educat"]]


# %%
Query = QueryTerms(corpus=corpus_full)

# %%
query_df = Query.find_matches(USER_TERMS, return_only_matches=True)

# %%
# CB.add_company_data(query_df[query_df["['parent']"]==True], columns=['name', 'long_description']).short_description.to_list()

# %%
