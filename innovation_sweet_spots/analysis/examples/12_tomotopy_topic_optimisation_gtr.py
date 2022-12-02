# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example of optimising number of topics for Gateway to Research project abstracts

# %% [markdown]
# ## Library imports

# %%
from innovation_sweet_spots.getters.gtr_2022 import get_gtr_file
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis.lda_modelling_utils import (
    topic_model_param_search,
    plot_param_search_results,
    highest_coherence_model_params,
)
import tomotopy as tp
import spacy
from tqdm import tqdm

# %% [markdown]
# ## Load data

# %%
# Load GtR project abstracts
gtr_project_abstracts = get_gtr_file(filename="gtr_projects-projects.json")[
    ["id", "abstractText"]
].rename(columns={"abstractText": "abstract"})

# %%
# View GtR project abtracts
gtr_project_abstracts.head(5)

# %%
# For testing, reduce data by sampling 5,000 records
gtr_project_abstracts = gtr_project_abstracts.sample(n=5_000, random_state=1)

# %% [markdown]
# ## Prepare text and create Corpus

# %%
# Create a corpus
corpus = tp.utils.Corpus()
# Load english Spacy components
nlp = spacy.load("en_core_web_sm")

# Add tokenised abstracts to corpus
for abstract in tqdm(
    gtr_project_abstracts.abstract.values, desc="Adding abstracts to corpus"
):
    tokenized_abstract = [token.text for token in nlp.tokenizer(abstract)]
    corpus.add_doc(tokenized_abstract)

# %%
# Set training parameters
N_TOPICS = [20, 40, 80]  # Number of topics to search through in param_search
ITERATIONS = 200
N_ITERS_LOGGING = 20

# %%
# Train topic models with specified parameters above
search_results = topic_model_param_search(
    n_topics=N_TOPICS,
    max_iterations=ITERATIONS,
    n_iters_logging=N_ITERS_LOGGING,
    corpus=corpus,
)

# %%
# View results
search_results

# %%
# Plot coherence score results for different iterations and n_topics
plot_param_search_results(search_results)

# %%
# Find highest coherence_score params
highest_coherence_model_params(search_results)
