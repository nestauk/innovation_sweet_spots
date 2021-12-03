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
# # Detect and analyse collocations

# %% [markdown]
# - [x] Identify key related terms using PMI and normalised rank
# - [x] Aggregate across set of terms
# - [ ] Analyse changes over time
# - [ ] Plot various measures of relevance over time

# %% [markdown]
# ## 1. Import Dependencies

# %%
import pandas as pd
import spacy
import pickle
from collections import defaultdict

# %%
from innovation_sweet_spots.utils import pd_collocation_utils as cu
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH

# %%
# OUTPUTS_CACHE = OUTPUT_DATA_PATH / ".cache"
DISC_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"
DISC_LOOKUPS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_lookups"
DISC_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
nlp = spacy.load("en_core_web_sm")

# %%
search_terms = ['heat pump', 'heat pumps']

# %% [markdown]
# ## 2. Detect collocations

# %%
# Read in outputs.

#article_text = pd.read_csv(os.path.join(DISC_OUTPUTS_DIR, 'article_text_hydrogen.csv'))

with open(DISC_OUTPUTS_DIR / 'sentence_records_hp.pkl', "rb") as infile:
        sentence_records = pickle.load(infile)

# %%
sentence_collection_df = pd.DataFrame(sentence_records)
sentence_collection_df.columns = ['sentence', 'id', 'year']
sentences_by_year = {y: v for y, v in sentence_collection_df.groupby('year')}

# %%
related_terms = defaultdict(dict)
normalised_ranks = defaultdict(dict)

for year, sentences in sentences_by_year.items():
    for term in search_terms:
        print(year, term)
        key_terms, normalised_rank = cu.get_key_terms(term, 
                                                      sentences['sentence'], 
                                                      nlp,
                                                      mentions_threshold = 1, 
                                                      token_range = (1,3))

        related_terms[year][term] = list(key_terms.items())
        normalised_ranks[year][term] = list(normalised_rank.items())

# %% [markdown]
# ## 3. Aggregate for a set of terms

# %%
combined_pmi= cu.combine_pmi(related_terms, search_terms)
combined_ranks = cu.combine_ranks(normalised_ranks, search_terms)

# %%
combined_pmi_dict = defaultdict(dict)
for year in combined_pmi:
    for term in combined_pmi[year]:
        combined_pmi_dict[year][term[0]] = term[1]

# %%
# Dictionary mapping search terms to frequency of mentions, normalised rank and pmi in a given year.
pmi_inters_ranks = defaultdict(dict)
for year in combined_ranks:
    for term in combined_ranks[year]:
        if term[0] in combined_pmi_dict[year]:
            pmi_inters_ranks[year][term[0]] = (term[1], combined_ranks[year][term], combined_pmi_dict[year][term[0]])

# %%
pmi_inters_ranks['2020']

# %%
# Aggregate into one long dataframe.
agg_pmi = cu.agg_combined_pmi_rank(pmi_inters_ranks)

# %%
# Preprocess for further analysis of changes over time.
# This spreadsheet can be used to identify terms with the largest change in collocation metrics.
# The spreadsheet shows for each term: year of first mention, total number of years with mentions, 
# standard deviation of the rank and mean pmi
agg_terms = cu.analyse_rank_pmi_over_time(agg_pmi)

# %%
# Save to disc to explore separately.
agg_pmi.to_csv(DISC_OUTPUTS_DIR / 'agg_terms_long_hp.csv', index = False)
agg_terms.to_csv(DISC_OUTPUTS_DIR / 'agg_terms_stats_hp.csv', index = False)
