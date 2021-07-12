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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Keyword expansion

# %%
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.utils.io as iss_io
import innovation_sweet_spots.utils.text_cleaning_utils as iss_text
import innovation_sweet_spots.analysis.text_analysis as iss_text_analysis
from innovation_sweet_spots.getters import gtr, misc

import numpy as np
import pandas as pd
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import gensim.models.phrases as phrases

DATA_OUTPUTS = PROJECT_DIR / "outputs/data"

# %%
from gensim.models import Word2Vec

# %%
import innovation_sweet_spots.analysis.green_docs as iss_green
import importlib

# %%
gtr_projects = gtr.get_gtr_projects()

# %%
import innovation_sweet_spots.utils.text_pre_processing as iss_preproc

# %% [markdown]
# ## Word2Vec

# %%
importlib.reload(iss_green)
importlib.reload(gtr)

# %%

# %%
green_keywords = iss_green.get_green_keywords(clean=True)
green_projects = iss_green.find_green_gtr_projects(green_keywords)
green_companies = iss_green.find_green_cb_companies()

# %% [markdown]
# ### Good quality phrases

# %%
green_project_texts = iss.create_documents_from_dataframe(
    green_projects,
    columns=["title", "abstractText", "techAbstractText"],
    preprocessor=(lambda x: x),
)
green_company_texts = iss.create_documents_from_dataframe(
    green_companies,
    columns=["short_description", "long_description"],
    preprocessor=(lambda x: x),
)
green_texts = green_project_texts + green_company_texts

# %%
corpus, ngram_phraser = iss_preproc.pre_process_corpus(green_texts)

# %%
fpath = PROJECT_DIR / "outputs/models/bigram_phraser_gtr_cb_v1.p"

# %%
pickle.dump(ngram_phraser, open(fpath, "wb"))

# %%
ngram_phraser = pickle.load(ngram_phraser, open(fpath, "rb"))

# %% [markdown]
# ## Word2vec expansions

# %% [markdown]
# ### Train the model

# %%
corpus_gtr = corpus[0 : len(green_project_texts)]
corpus_cb = corpus[len(green_project_texts) :]

# %%
iss_io.save_list_of_terms(
    green_projects.project_id.to_list(),
    PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised_ids.txt",
)
iss_io.save_list_of_terms(
    green_companies.id.to_list(),
    PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised_ids.txt",
)

# %%
pickle.dump(
    corpus_gtr, open(PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p", "wb")
)
pickle.dump(
    corpus_cb, open(PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised.p", "wb")
)

# %%
# corpus_gtr[999]

# %%
word2vec_gtr = Word2Vec(
    sentences=corpus_gtr,
    vector_size=300,
    window=10,
    min_count=5,
    workers=4,
    sg=0,
    epochs=50,
    seed=111,
)

# %%
word2vec_cb = Word2Vec(
    sentences=corpus_cb,
    vector_size=300,
    window=10,
    min_count=5,
    workers=4,
    sg=0,
    epochs=50,
    seed=111,
)

# %%
fpath = PROJECT_DIR / "outputs/models/word2vec_gtr_v1.p"
pickle.dump(word2vec_gtr, open(fpath, "wb"))
fpath = PROJECT_DIR / "outputs/models/word2vec_cb_v1.p"
pickle.dump(word2vec_cb, open(fpath, "wb"))

# %%

# %% [markdown]
# ### Matching keywords

# %%
green_keywords_initial = iss_green.get_green_keywords(clean=False)

# %%
nlp = iss_preproc.setup_spacy_model(iss_preproc.DEF_LANGUAGE_MODEL)


# %%
def expand_keywords(word2vec_model, initial_keywords, ngram_phraser, topn=10):
    """Given a word2vec model, expand the initial keywords"""
    vocab = list(word2vec_model.wv.key_to_index.keys())
    best_matches = pd.DataFrame()
    logging.info(f"Expanding {len(initial_keywords)} terms")
    for keyword in initial_keywords:
        # Tokenise the key term if it's not present in the vocabulary
        keyword_snake = "_".join(keyword.split())
        if keyword_snake in vocab:
            keyword_list = [keyword_snake]
        else:
            keyword_list = [
                term
                for term in iss_preproc.ngrammer(
                    keyword, nlp=nlp, ngram_phraser=ngram_phraser
                )
                if term in vocab
            ]
        if len(keyword_list) > 0:
            top_similar = word2vec_model.wv.most_similar(
                positive=keyword_list, topn=topn
            )
            df = pd.DataFrame(top_similar, columns=["expanded_term", "score"])
            df["seed_term"] = keyword
            best_matches = best_matches.append(df, ignore_index=True)

    best_matches.sort_values("score", ascending=False).drop_duplicates(
        "expanded_term", keep="first"
    )
    # Remove keywords that already exist
    best_matches = best_matches[
        best_matches.expanded_term.isin(
            ["_".join(keyword.split()) for keyword in initial_keywords]
        )
        == False
    ]

    logging.info(f"Expanded to additional {len(best_matches)} terms")
    return best_matches[["seed_term", "expanded_term", "score"]]


# %%
expanded_keywords_gtr = expand_keywords(
    word2vec_gtr, green_keywords_initial, ngram_phraser
)
expanded_keywords_gtr["source"] = "GTR"
expanded_keywords_cb = expand_keywords(
    word2vec_cb, green_keywords_initial, ngram_phraser
)
expanded_keywords_cb["source"] = "CB"
expanded_keywords = (
    pd.concat([expanded_keywords_gtr, expanded_keywords_cb])
    .drop_duplicates("expanded_term", keep="first")
    .reset_index()
)


# %%
# word2vec_cb.wv.most_similar(positive=['carbon_emissions'], topn=10)

# %%
expanded_keywords.to_csv(
    PROJECT_DIR / "outputs/data/aux/green_keywords_expanded_raw.csv", index=False
)

# %% [markdown]
# ### Find the most similar

# %%
green_keywords = iss_green.get_green_keywords(clean=False)

# %%
# [bigrammer(keyword) for keyword in green_keywords]

# %% [markdown]
# ### Inspect

# %%
# gensim.parsing.preprocessing.strip_tags("Uses for the &quot; entity in HTML - Stack Overflow")

# %% [markdown]
# ### Finalise the green document corpus

# %%
import innovation_sweet_spots.utils.text_cleaning_utils as iss_text

# %%
