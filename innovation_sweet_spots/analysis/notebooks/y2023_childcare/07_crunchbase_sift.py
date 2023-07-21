# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Search in Crunchbase for child care companies
#
# **Industry search**
# - Using industry tags, select potential candidates
# - Using filtering terms, narrow down potential candidates
#     - (One versus two mentions of filtering terms)
# - Investigate other ways to filter
#     - Number of companies with at least 1 deal or funding amount data
#     - Number of companies from Europe + US + Australia/NZ
#     - Using non-filter categories (ie, child, family etc)
#
# **Keyword search**
# - Using taxonomy keywords, select potential candidates
# - Using filtering terms, narrow down potential candidates
# - Investgate other ways to filter
#
# **Clean up**
# - If a company is in multiple categories, try to minimise that
#     - eg, if a company is in a more generic and specific categories, remove from generic

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_full_crunchbase_corpus
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.utils.google_sheets as gs
from innovation_sweet_spots.analysis.query_categories import query_cb_categories
import utils
from ast import literal_eval
import pandas as pd
from innovation_sweet_spots.utils import cluster_analysis_utils

# Calculate embeddings
from innovation_sweet_spots.analysis import wrangling_utils as wu
from innovation_sweet_spots.utils.embeddings_utils import QueryEmbeddings
from innovation_sweet_spots.utils.embeddings_utils import Vectors


import importlib

importlib.reload(utils)

# %%
CB = CrunchbaseWrangler()

# %%
# Load taxonomy and the associated keywords
taxonomy_df = gs.download_google_sheet(utils.AFS_GOOGLE_SHEET_ID, "taxonomy")
# Processed keywords
keywords_df = utils.process_keywords(taxonomy_df, "keywords", add_plurals=True)

# Load aligned categories
aligned_categories_df = gs.download_google_sheet(
    utils.AFS_GOOGLE_SHEET_ID, "taxonomy_alignment"
).astype({"need_filtering": int})

# Load initial list of companies
initial_list_df = gs.download_google_sheet(utils.AFS_GOOGLE_SHEET_ID, "initial_list")

# Load a table with processed company descriptions
processed_texts = pd.read_csv(
    utils.COMPANY_DESCRIPTIONS_PATH, names=["id", "name", "description"]
)

# %% [markdown]
# ## Filtering set
#
# ### Filtering set based on "safe" industry tags

# %%
# Companies that belong to non-filtered industries
non_filtered_industries = aligned_categories_df.query(
    'need_filtering == 0 and source == "crunchbase_categories"'
).category.to_list()
# Log the names of non filtered industries
logging.info(f"Non-filtered industries: {non_filtered_industries}")
# Get the ids of companies that belong to non-filtered industries
children_industry_ids = utils.query_categories(non_filtered_industries, CB)


# %% [markdown]
# ### Filtering set based on selected filtering terms

# %%
# Process the company descriptions for keyword search
processed_texts["description_processed"] = (
    " "
    + (
        processed_texts["description"]
        # lower case
        .str.lower()
        # replace punctuation with spaces
        .str.replace("[^\w\s]", " ")
        # replace multiple spaces with single space
        .str.replace("\s+", " ")
        # strip leading and trailing spaces
        .str.strip()
    )
    + " "
)

# Load the companies into an instance of QueryTerms
corpus_full = dict(zip(processed_texts.id, processed_texts.description_processed))
Query = QueryTerms(corpus_full)
Query.verbose = False

# Select the filtering keywords
filtering_keywords = keywords_df.query(
    'theme == "Filtering terms"'
).keywords_processed.to_list()

# Query the companies for the filtering keywords
filtered_company_ids = set(
    Query.find_matches(filtering_keywords, return_only_matches=True).id.to_list()
)

# %% [markdown]
# ### Create the final filtering set

# %%
# Combine both filtering sets
filter_ok_ids = children_industry_ids.union(filtered_company_ids)
logging.info(f"Number of companies after filtering: {len(filter_ok_ids)}")

# %% [markdown]
# ## Industry search

# %%
# Industries that we wish to search for
search_industries_df = aligned_categories_df.query('source == "crunchbase_categories"')
search_industries = search_industries_df.category.to_list()
logging.info(f"Industries to search for: {search_industries}")

# Query the companies for the industries
df_hits = query_cb_categories(search_industries, CB, return_only_matches=True).query(
    "id in @filter_ok_ids"
)

# Process the industry search results
df_hits_melt = (
    pd.melt(df_hits.drop("any_category", axis=1), id_vars="id")
    .query("value == True")
    .merge(
        aligned_categories_df.query('source == "crunchbase_categories"')[
            ["category", "theme", "subtheme", "source"]
        ],
        left_on="variable",
        right_on="category",
        how="left",
    )
    .drop(["variable", "value"], axis=1)
)

# Log the number of results
logging.info(f"Number of hits: {len(df_hits_melt)}")

# %% [markdown]
# ## Keyword search

# %%
# Select keywords
keywords_df_cats = keywords_df.query('theme != "Filtering terms"')
keywords_list = keywords_df_cats.keywords_processed.to_list()

# Query the companies for the keywords
df_keyword_hits = Query.find_matches(keywords_list, return_only_matches=True).query(
    "id in @filter_ok_ids"
)

# Process the keyword search results
df_keywords_melt = (
    # change the format of the results
    pd.melt(df_keyword_hits.drop("has_any_terms", axis=1), id_vars="id")
    # select only the keyword hits
    .query("value == True")
    # change to string, so that we can merge with the keywords table
    .astype({"variable": str})
    .merge(
        keywords_df_cats[["keywords_processed", "theme", "subtheme"]].astype(
            {"keywords_processed": str}
        ),
        left_on="variable",
        right_on="keywords_processed",
        how="left",
    )
    # clean up
    .drop(["variable", "value"], axis=1)
    .drop_duplicates(["id", "keywords_processed"])
    # turn the keywords back to list
    .assign(
        keywords_processed_list=lambda df: df.keywords_processed.apply(
            lambda x: literal_eval(x)
        )
    )
)


# %% [markdown]
# ### Characterise the keyword search results a bit

# %%
# group by id, theme and subtheme and aggregate the lists of keywords
df_keywords_melt_grouped = (
    df_keywords_melt.groupby(["id", "theme", "subtheme"])
    .agg({"keywords_processed_list": "sum"})
    .reset_index()
)

# Check most common keyword search results by theme
df_keywords_melt_grouped.groupby("theme").agg(counts=("id", "count")).reset_index()

# %%
# Check most common keyword search results by subtheme
df_keywords_melt_grouped.groupby(["theme", "subtheme"]).agg(
    counts=("id", "count")
).reset_index()

# %% [markdown]
# ## Combine industry and keyword search results

# %%
# Combine hits from categories and keywords
df_all_hits = pd.concat(
    [df_hits_melt, df_keywords_melt_grouped.assign(source="crunchbase_keywords")]
)

# Make a deduplicated list of companies, and combine the multiple rows for one company into one row
childcare_candidates = (
    df_all_hits.groupby("id")
    .agg(
        themes=("theme", "unique"),
        subthemes=("subtheme", "unique"),
        category=("category", "unique"),
        sources=("source", "unique"),
        keywords_processed_list=("keywords_processed_list", "sum"),
    )
    .reset_index()
)
childcare_candidates

# %%
# Characterise all hits
df_all_hits.groupby(["theme", "subtheme"]).agg(counts=("id", "count")).reset_index()

# %% [markdown]
# ## Narrow down to a shortlist
# - Companies in Europe, US and Australia
# - Companies with at least 1 deal or funding amount data
# - Companies that are not closed

# %%
logging.info(f"Number of orgs before filtering: {len(childcare_candidates)}")

# Filter by country
childcare_candidates_filtered = (
    childcare_candidates.merge(
        CB.cb_organisations[["id", "name", "country", "status"]], how="left", on="id"
    )
    .query("country in @utils.list_of_select_countries")
    .query('status != "closed"')
)

logging.info(
    f"Number of orgs after filtering by country: {len(childcare_candidates_filtered)}"
)

# Filter by funding
org_ids_with_funding = CB.get_funding_rounds(childcare_candidates_filtered)
childcare_candidates_filtered = childcare_candidates_filtered.query(
    "id in @org_ids_with_funding.org_id.to_list()"
)

logging.info(
    f"Number of orgs after filtering by funding: {len(childcare_candidates_filtered)}"
)

# %% [markdown]
# # Merge with the initial list of companies
#
# Merge with the initial list (including companies from other sources)

# %%
# Select only hits that are in the filtered list
df_all_hits_filtered = df_all_hits.query(
    "id in @childcare_candidates_filtered.id.to_list()"
)

# Align the categories to our taxonomy
initial_list_df_aligned = (
    initial_list_df[["cb_id", "company_name", "homepage_url", "source", "category"]]
    .query('source != "crunchbase_categories"')
    .merge(
        aligned_categories_df[["source", "category", "theme", "subtheme"]],
        on=["source", "category"],
        how="left",
    )
    .query('theme != "-"')
    .rename(columns={"cb_id": "id", "company_name": "name"})
)

# Combine the hits from intial list and this search
combined_hits = pd.concat(
    [
        initial_list_df_aligned,
        df_all_hits_filtered.merge(
            CB.cb_organisations[["id", "name", "homepage_url"]], on="id", how="left"
        ),
    ],
    ignore_index=True,
)

# %% [markdown]
# # Add helpful data
#
# ## Generate keywords for each company

# %%
childcare_candidates_final_df = (
    combined_hits.groupby("id")
    .agg(
        themes=("theme", "unique"),
        subthemes=("subtheme", "unique"),
        category=("category", "unique"),
        sources=("source", "unique"),
        keywords_processed_list=("keywords_processed_list", "sum"),
    )
    # Add crunchbase data
    .merge(
        CB.cb_organisations[
            [
                "id",
                "name",
                "homepage_url",
                "country",
                "status",
                "cb_url",
                "country",
                "region",
                "city",
                "rank",
                "short_description",
                "long_description",
                "rank",
            ]
        ],
        on="id",
        how="left",
    )
    # Add description text
    .merge(processed_texts[["id", "description_processed"]], how="left", on="id")
    .merge(
        processed_texts[["id", "description"]].rename(
            columns={"description": "description_raw"}
        ),
        how="left",
        on="id",
    )
    .fillna("")
    .reset_index()
)

# Add keywords from the description
company_keywords = cluster_analysis_utils.cluster_keywords(
    childcare_candidates_final_df.description_processed.to_list(),
    list(range(len(childcare_candidates_final_df))),
)

childcare_candidates_final_df["company_keywords"] = list(company_keywords.values())

# Add investors
childcare_candidate_investors = (
    CB.get_organisation_investors(childcare_candidates_final_df)
    .groupby("org_id")
    .agg(investors=("investor_name", "unique"))
    .reset_index()
)

childcare_candidates_final_df = childcare_candidates_final_df.merge(
    childcare_candidate_investors.rename(columns={"org_id": "id"}), how="left", on="id"
)


# %%
# Funding data
funding_rounds = CB.get_funding_rounds(
    (
        childcare_candidates_final_df[
            childcare_candidates_final_df.id.notnull()
        ].drop_duplicates("id")
    ),
    org_id_column="id",
)

# Rename columns
childcare_candidates_final_df.rename(
    columns={"id": "cb_id", "name": "company_name"},
    inplace=True,
)
# merge all the funding dataframes to company_list
company_list_funding = (
    childcare_candidates_final_df.merge(
        utils.get_last_rounds(funding_rounds), on="cb_id", how="left"
    )
    .merge(utils.get_last_rounds_since_2020(funding_rounds), on="cb_id", how="left")
    .merge(utils.get_total_funding(funding_rounds), on="cb_id", how="left")
    .merge(
        initial_list_df[
            [
                "company_name",
                "homepage_url",
                "relevancy_comment",
                "other_comments",
                "relevant",
            ]
        ],
        on=["company_name", "homepage_url"],
        how="left",
    )
    # drop null rows
    .dropna(subset=["company_name"])
)

company_list_funding = utils.investibility_indicator(company_list_funding)

# %% [markdown]
# # Embeddings and quick clustering
#

# %%
# For clustering, select only companies that have a description raw longer than 100 characters
df_companies = (
    company_list_funding.drop_duplicates(subset=["cb_id"])
    # select only companies that have a description raw longer than 100 characters
    .loc[lambda x: x.description_raw.str.len() > 100]
)

logging.info(
    f"Number of orgs after filtering by description length: {len(df_companies)}"
)

# %%
# Define constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

EMBEDDINGS_DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "test_childcare_companies"

# Instantiate the model
childcare_vectors = Vectors(
    model_name=EMBEDDING_MODEL,
    vector_ids=None,
    filename=FILENAME,
    folder=EMBEDDINGS_DIR,
)
# Make vectors
childcare_vectors.generate_new_vectors(
    new_document_ids=df_companies.cb_id.values,
    texts=df_companies.description_raw.values,
)
# Save vectors
childcare_vectors.save_vectors()


# %%
# Define K-Means search parameters
kmeans_search_params = {"n_clusters": [10, 20, 25, 30, 35, 40], "init": ["k-means++"]}

# Parameter grid search using K-Means
kmeans_search_results = cluster_analysis_utils.kmeans_param_grid_search(
    vectors=childcare_vectors.vectors,
    search_params=kmeans_search_params,
    random_seeds=[42, 43, 44],
)

# %%
optimal_kmeans_params = cluster_analysis_utils.highest_silhouette_model_params(
    kmeans_search_results
)
print(optimal_kmeans_params)
kmeans_search_results

# %%
param_dict = literal_eval(optimal_kmeans_params)
param_dict["random_state"] = 42
param_dict["n_clusters"] = 25

optimal_labels = cluster_analysis_utils.kmeans_clustering(
    childcare_vectors.vectors, param_dict
)

# %%
df_companies["cluster_label"] = optimal_labels

# Add keywords from the description
optimal_cluster_keywords = cluster_analysis_utils.cluster_keywords(
    df_companies.description_processed.to_list(), df_companies.cluster_label.to_list()
)

df_companies["cluster_keywords"] = df_companies.cluster_label.map(
    optimal_cluster_keywords
)


# %%
pd.set_option("display.max_colwidth", 300)
cluster_label_valid_df = (
    pd.DataFrame(
        data={
            "cluster_label": list(optimal_cluster_keywords.keys()),
            "cluster_keywords": list(optimal_cluster_keywords.values()),
        }
    )
    .sort_values("cluster_label")
    .reset_index(drop=True)
)
# dictionary of cluster label (string with integer from 0 to 24) and values are either 'likely relevant' or 'noise'
quick_validation = {
    "0": "relevant",
    "1": "noise",
    "2": "noise",
    "3": "noise",
    "4": "noise",
    "5": "noise",
    "6": "noise",
    "7": "noise",
    "8": "relevant",
    "9": "relevant",
    "10": "noise",
    "11": "not sure",
    "12": "not sure",
    "13": "relevant",
    "14": "relevant",
    "15": "not sure",
    "16": "noise",
    "17": "noise",
    "18": "noise",
    "19": "relevant",
    "20": "relevant",
    "21": "noise",
    "22": "noise",
    "23": "noise",
    "24": "noise",
}

df_companies["cluster_relevance"] = df_companies.cluster_label.astype(str).map(
    quick_validation
)

# %%
cluster_label_valid_df[
    "cluster_relevance"
] = cluster_label_valid_df.cluster_label.astype(str).map(quick_validation)
cluster_label_valid_df

# %%
vectors_2d, fig = cluster_analysis_utils.cluster_visualisation(
    # childcare_vectors.vectors,
    childcare_vectors.vectors,
    optimal_labels,
    # Add short abstracts to the visualisation
    extra_data=(
        df_companies[
            [
                "company_name",
                "description_raw",
                "company_keywords",
                "sources",
                "category",
                "subthemes",
                "homepage_url",
            ]
        ]
        # .assign(abstract=lambda df: df.abstract.apply(lambda x: str(x)[0:300] + '...'))
    ),
)

# %%
vectors_2d, fig = cluster_analysis_utils.cluster_visualisation(
    # childcare_vectors.vectors,
    vectors_2d,
    (
        df_companies["cluster_label"].astype(str)
        + ": "
        + df_companies["cluster_keywords"].astype(str)
    ).to_list(),
    # Add short abstracts to the visualisation
    extra_data=(
        df_companies[
            [
                "cluster_keywords",
                "cluster_relevance",
                "company_name",
                "description_raw",
                "company_keywords",
                "sources",
                "category",
                "subthemes",
                "homepage_url",
                "cb_url",
            ]
        ]
        # .assign(abstract=lambda df: df.abstract.apply(lambda x: str(x)[0:300] + '...'))
    ),
)

# %%
# save altair figure as html
# fig.save(PROJECT_DIR / 'outputs/2023_childcare/interim/list_v2_visual.html')

# %%
# add url encoding to the altair figure
fig_ = fig.encode(href="cb_url:N")

# %%
fig

# %%
# export altair to html
fig_.save(PROJECT_DIR / "outputs/2023_childcare/interim/list_v2_visual.html")

# %% [markdown]
# # Upload the data to Google Sheet

# %%
company_list_funding_clustered = company_list_funding.merge(
    df_companies[["cb_id", "cluster_label", "cluster_keywords", "cluster_relevance"]],
    on="cb_id",
    how="left",
)

# %%
# change the order of columns
company_list_export = company_list_funding_clustered.copy()[
    [
        "company_name",
        "homepage_url",
        "cluster_label",
        "cluster_keywords",
        "cluster_relevance",
        "cb_id",
        "cb_url",
        "sources",
        "themes",
        "subthemes",
        "company_keywords",
        "short_description",
        "long_description",
        "country",
        "region",
        "city",
        "rank",
        "rank",
        "investors",
        "last_round_date",
        "investment_type",
        "last_round_gbp",
        "last_round_usd",
        "last_valuation_usd",
        "last_round_investor_count",
        "deal_url",
        "funding_since_2020_gbp",
        "funding_rounds_since_2020",
        "total_funding_gbp",
        "relevancy_comment",
        "other_comments",
        "relevant",
        "investible",
    ]
]
company_list_export = company_list_export.fillna(
    {
        "cb_url": "n/a",
        "relevant": "not evaluated",
        "cluster_label": "not clustered",
        "cluster_keywords": "not clustered",
        "cluster_relevance": "not clustered",
    }
)

# %%
# Necessary for uploading to work
save_path = PROJECT_DIR / "outputs/2023_childcare/interim/v2_list_07Feb2023.csv"
company_list_export.to_csv(save_path, index=False)
company_list_export = pd.read_csv(save_path)

# Save to Google Sheet
gs.upload_to_google_sheet(
    company_list_export,
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
    wks_name="list_v2",
)
