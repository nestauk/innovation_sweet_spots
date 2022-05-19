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

# %% [markdown]
# # Company analysis
#
# For any selection or organisations, prepares the following report:
#
# - Total investment across years
# - Number of deals across years
# - Types of investment (NB: needs simpler reporting of deal types)
# - Investment by countries (top countries)
# - Top UK cities
# - Fraction of digital among these companies
# - Investment trends by digital category
# - Baselines
# - Select examples

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots.analysis.notebooks.parenting import utils
import innovation_sweet_spots.analysis.analysis_utils as au
import innovation_sweet_spots.utils.plotting_utils as pu

import importlib

importlib.reload(utils)
importlib.reload(au)
importlib.reload(pu)

# %%
from innovation_sweet_spots import PROJECT_DIR

# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/finals/parenting/cb_companies"

# %%
import pandas as pd

# %%
CB = CrunchbaseWrangler()


# %%
def select_by_role(cb_orgs: pd.DataFrame, role: str):
    """
    Select companies that have the specified role.
    Roles can be 'investor', 'company', or 'both'
    """
    all_roles = cb_orgs.roles.copy().fillna("")
    if role != "both":
        return cb_orgs[all_roles.str.contains(role)]
    else:
        return cb_orgs[
            all_roles.str.contains("investor") & all_roles.str.contains("company")
        ]


# %%
pu.test_chart()

# %%
check_columns = ["name", "short_description", "long_description"]

# %% [markdown]
# # Get reviewed companies

# %%
inputs_path = PROJECT_DIR / "outputs/finals/parenting/cb_companies/reviewed"

# %%
reviewed_df_parenting = pd.read_csv(
    inputs_path
    / "cb_companies_parenting_v2022_04_27 - cb_companies_parenting_v2022_04_27.csv"
)
reviewed_df_child_ed = pd.read_csv(
    inputs_path
    / "cb_companies_child_ed_v2022_04_27 - cb_companies_child_ed_v2022_04_27.csv"
)

# %%
# Select the companies with 'relevant'
reviewed_df_child_ed.info()

# %%
companies_parenting_df = reviewed_df_parenting.query('relevancy == "relevant"')
companies_child_ed_df = reviewed_df_child_ed.query(
    'relevancy == "relevant" or comment == "potentially relevant"'
)

# %%
companies_ids = set(companies_parenting_df.id.to_list()).union(
    set(companies_child_ed_df.id.to_list())
)

# %%
len(companies_ids)

# %% [markdown]
# # Analyse parenting companies

# %% [markdown]
# ## Selection

# %%
# importlib.reload(utils);
# cb_orgs_parenting = (
#     CB.get_companies_in_industries(utils.PARENT_INDUSTRIES)
# )

# %%
cb_orgs = CB.cb_organisations.query("id in @companies_ids")

# %% [markdown]
# ## Analysis

# %%
cb_companies = cb_orgs.pipe(select_by_role, "company")
cb_companies_with_funds = au.get_companies_with_funds(cb_companies)

# %%
len(cb_companies_with_funds)

# %%
funding_df = CB.get_funding_rounds(cb_companies_with_funds)
funding_ts = au.cb_get_all_timeseries(
    cb_companies_with_funds, funding_df, "year", 2010, 2021
)

# %%
# funding_df.head(3)

# %%
funding_ts.head(3)

# %%
pu.time_series(funding_ts, y_column="raised_amount_gbp_total")

# %%
funding_ts.head(2)

# %%
importlib.reload(pu)
pu.cb_investments_barplot(
    funding_ts,
    y_column="raised_amount_gbp_total",
    y_label="Raised investment (1000s GBP)",
    x_label="Year",
)

# %%
pu.cb_investments_barplot(
    funding_ts, y_column="no_of_rounds", y_label="Number of deals", x_label="Year"
)

# %%
pu.cb_investments_barplot(
    funding_ts,
    y_column="no_of_orgs_founded",
    y_label="Number of new companies",
    x_label="Year",
)

# %%
importlib.reload(pu)
pu.cb_deal_types(funding_df, simpler_types=True)

# %%
importlib.reload(au)
funding_by_country = au.cb_funding_by_geo(cb_orgs, funding_df)
funding_by_city = au.cb_funding_by_geo(cb_orgs, funding_df, "org_city")

# %%
importlib.reload(pu)
pu.cb_top_geographies(
    funding_by_country,
    "no_of_rounds",
    value_label="Number of deals",
)

# %%
importlib.reload(pu)
pu.cb_top_geographies(
    funding_by_city,
    value_column="no_of_rounds",
    value_label="Number of deals",
    category_column="org_city",
)

# %%
importlib.reload(pu)
pu.cb_top_geographies(
    funding_by_country,
    value_column="raised_amount_gbp",
    value_label="Raised amount (£1000s)",
)

# %%
importlib.reload(au)
funding_geo_ts = au.cb_get_timeseries_by_geo(
    cb_companies_with_funds,
    funding_df,
    geographies=["United States", "United Kingdom", "China", "Germany"],
    period="year",
    min_year=2010,
    max_year=2021,
)

# %%
importlib.reload(pu)
pu.time_series_by_category(
    funding_geo_ts,
    value_column="no_of_rounds",
    #     value_label = 'Raised amount (£1000s)'
)

# %%
importlib.reload(pu)
pu.time_series_by_category(
    funding_geo_ts,
    value_column="raised_amount_gbp_total",
    #     value_label = 'Raised amount (£1000s)'
)

# %%
importlib.reload(pu)
pu.time_series_by_category(
    funding_geo_ts,
    value_column="raised_amount_gbp_total",
    #     value_label = 'Raised amount (£1000s)'
)

# %%
importlib.reload(pu)
funding_by_city = au.cb_funding_by_geo(
    cb_orgs.query('country == "United Kingdom"'), funding_df, "org_city"
)
pu.cb_top_geographies(
    funding_by_city,
    value_column="no_of_rounds",
    value_label="Number of deals",
    category_column="org_city",
)

# %%
importlib.reload(au)

pu.cb_top_geographies(
    au.cb_companies_by_geo(cb_companies),
    value_column="no_of_companies",
    value_label="Number of companies",
    category_column="country",
)

# %%
importlib.reload(au)

pu.cb_top_geographies(
    au.cb_companies_by_geo(
        cb_companies.query('country == "United Kingdom"'), geo_entity="city"
    ),
    value_column="no_of_companies",
    value_label="Number of companies",
    category_column="city",
)


# %% [markdown]
# ## Digital technologies
# - Select companies in industries ("digital")
# - Get the number of companies founded by year (by industry)
# - Get the number of deals by year (by industry)
# - Long term trends (5 year trend)
# - Short term trends (2020 vs 2021)

# %%
# Which companies are in digital
importlib.reload(utils)
digital = utils.get_digital_companies(cb_companies, CB)

# %%
importlib.reload(utils)
utils.digital_proportion(cb_companies, digital)

# %%
importlib.reload(utils)
utils.digital_proportion(cb_companies, digital, since=2011)

# %%
digital_ids = digital.id.to_list()
cb_companies.query("id in @digital_ids").total_funding_usd.sum()

# %%
importlib.reload(au)
top_industries = au.cb_top_industries(digital, CB)

# %%
top_industries.query("industry in @utils.DIGITAL_INDUSTRIES").head(50)

# %%
importlib.reload(utils)
digital_fraction_ts = utils.digital_proportion_ts(cb_companies, digital, 1998, 2021)

# %%
importlib.reload(pu)
pu.cb_investments_barplot(
    digital_fraction_ts,
    y_column="digital_fraction",
    x_label="Time period",
)

# %%
importlib.reload(pu)
pu.time_series(digital_fraction_ts, y_column="digital_fraction")

# %%
# importlib.reload(pu)
# pu.time_series(digital_fraction_ts, y_column="digital_fraction")

# %%
importlib.reload(au)
(
    rounds_by_industry_ts,
    companies_by_industry_ts,
    investment_by_industry_ts,
) = au.investments_by_industry_ts(
    digital.drop("industry", axis=1),
    utils.DIGITAL_INDUSTRIES,
    #     ['software', 'apps'],
    CB,
    "no_of_rounds",
    2011,
    2021,
)


# %%
importlib.reload(au)
(
    rounds_by_group_ts,
    companies_by_group_ts,
    investment_by_group_ts,
) = au.investments_by_industry_ts(
    digital.drop("industry", axis=1),
    utils.DIGITAL_INDUSTRY_GROUPS,
    #     ['software', 'apps'],
    CB,
    "no_of_rounds",
    2011,
    2021,
    True,
)


# %%
rounds_by_group_ts

# %%
importlib.reload(au)
rounds_by_industry_ts_ma = au.ts_moving_average(rounds_by_industry_ts)

# %%
# pu.time_series(companies_by_industry_ts.reset_index(), y_column="advice")

# %%
cat = "data and analytics"
# cat = "apps"
pu.time_series(companies_by_group_ts.reset_index(), y_column=cat)

# %%
pu.time_series(investment_by_group_ts.reset_index(), y_column=cat)

# %%
pu.time_series(rounds_by_group_ts.reset_index(), y_column=cat)

# %%
# CB.industry_to_group['computer']

# %%
# CB.group_to_industries['artificial intelligence']

# %%
importlib.reload(au)
au.compare_years(investment_by_group_ts).query("reference_year!=0").sort_values(
    "growth", ascending=False
)


# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(rounds_by_group_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average number of deals")

# %%
# magnitude_growth.sort_values('growth', ascending=False).head(50)

# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(companies_by_group_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average number of new companies")

# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(investment_by_group_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average investment amount")

# %%
CB.group_to_industries["hardware"]

# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(investment_by_industry_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average investment amount")

# %%
magnitude_growth[-magnitude_growth.growth.isnull()].sort_values(
    ["growth", "magnitude"], ascending=False
).head(20)

# %%
cb_companies.info()

# %%
cb_companies_industries = df.merge(
    CB.get_company_industries(cb_companies, return_lists=True), on=["id", "name"]
)

# %%
cb_companies_industries

# %%
# importlib.reload(au)
# au.compare_years(investment_by_industry_ts).query("reference_year!=0").sort_values(
#     "growth", ascending=False
# )

# %%
df_funds[
    [
        "name",
        "short_description",
        "long_description",
        "homepage_url",
        "country",
        "founded_on",
        "total_funding_usd",
        "num_funding_rounds",
        "num_exits",
    ]
].sort_values("total_funding_usd", ascending=False).head(15)

# %%
# Add - millions or thousands
# Add - benchmarking
# Add time series for a country, and comparing countries

# %% [markdown]
# # Analyse children & education companies

# %% [markdown]
# ### Selection (slightly advanced)

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_full_crunchbase_corpus
import importlib
import innovation_sweet_spots.getters.preprocessed

importlib.reload(innovation_sweet_spots.getters.preprocessed)

# %%
from innovation_sweet_spots.analysis.query_categories import query_cb_categories

# %% [markdown]
# #### Select by industry

# %%
query_df_children = query_cb_categories(
    utils.CHILDREN_INDUSTRIES, CB, return_only_matches=True, verbose=False
)
query_df_education = query_cb_categories(
    utils.EDUCATION_INDUSTRIES, CB, return_only_matches=True, verbose=False
)
query_df_remove_industry = query_cb_categories(
    utils.INDUSTRIES_TO_REMOVE, CB, return_only_matches=True, verbose=False
)

# %%
children_industry_ids = set(query_df_children.id.to_list())
education_industry_ids = set(query_df_education.id.to_list())
remove_industry_ids = set(query_df_remove_industry.id.to_list())

children_education_ids = children_industry_ids.intersection(
    education_industry_ids
).difference(remove_industry_ids)

# %%
cb_orgs = CB.cb_organisations.query("id in @children_education_ids")
cb_companies = cb_orgs.pipe(select_by_role, "company")
cb_companies_with_funds = au.get_companies_with_funds(cb_companies)
print(len(cb_companies_with_funds))

# %%
len(children_education_ids), len(cb_companies), len(cb_companies_with_funds)

# %% [markdown]
# #### Select by keywords

# %%
corpus_full = get_full_crunchbase_corpus()

# %%
Query = QueryTerms(corpus=corpus_full)

# %%
importlib.reload(utils)
query_df_children = Query.find_matches(utils.CHILDREN_TERMS, return_only_matches=True)
query_df_learning_terms = Query.find_matches(
    utils.ALL_LEARNING_TERMS, return_only_matches=True
)

# %%
children_term_ids = set(query_df_children.id.to_list())
education_term_ids = set(query_df_education.id.to_list())

children_education_term_ids = children_term_ids.intersection(education_term_ids)

cb_orgs = CB.cb_organisations.query("id in @children_education_term_ids")
cb_companies_terms = cb_orgs.pipe(select_by_role, "company")
cb_companies_terms_with_funds = au.get_companies_with_funds(cb_companies_terms)
print(len(cb_companies_terms_with_funds))

# %%
len(children_education_term_ids), len(cb_companies_terms), len(
    cb_companies_terms_with_funds
)

# %% [markdown]
# #### Combine both selections

# %%
children_education_ids_all = children_education_ids.union(children_education_term_ids)
cb_orgs = CB.cb_organisations.query("id in @children_education_ids_all")
cb_companies = cb_orgs.pipe(select_by_role, "company")

# %%
cb_companies_with_funds = au.get_companies_with_funds(cb_companies)
print(len(cb_companies_with_funds))

# %%
len(children_education_ids), len(children_education_term_ids), len(
    children_education_ids_all
)

# %%
len(cb_companies), len(cb_companies_with_funds)

# %%
len(cb_companies_with_funds) / len(cb_companies)

# %% [markdown]
# #### Check the organisations

# %%
id_ = cb_companies_with_funds.iloc[9].id
# id_ = list(children_education_term_ids)[1]

# %%
pd.set_option("max_colwidth", 1000)
CB.cb_organisations.query("id == @id_")[check_columns]

# %% [markdown]
# ## Analysis

# %%
cb_companies_with_funds_ = cb_companies_with_funds.query("country != 'China'")
cb_companies_with_funds_ = cb_companies_with_funds  # .query("country != 'China'")

# %%
funding_df = CB.get_funding_rounds(cb_companies_with_funds_)
funding_ts = au.cb_get_all_timeseries(
    cb_companies_with_funds_, funding_df, "year", 2010, 2021
)

# %%
importlib.reload(pu)
pu.cb_investments_barplot(
    funding_ts,
    y_column="raised_amount_gbp_total",
    y_label="Raised investment (1000s GBP)",
    x_label="Year",
)

# %%
pu.cb_investments_barplot(
    funding_ts, y_column="no_of_rounds", y_label="Number of deals", x_label="Year"
)

# %%
pu.cb_investments_barplot(
    funding_ts,
    y_column="no_of_orgs_founded",
    y_label="Number of new companies",
    x_label="Year",
)

# %%
importlib.reload(pu)
pu.cb_deal_types(funding_df, simpler_types=True)

# %%
importlib.reload(au)
funding_by_country = au.cb_funding_by_geo(cb_companies_with_funds, funding_df)
funding_by_city = au.cb_funding_by_geo(cb_companies_with_funds, funding_df, "org_city")

# %%
importlib.reload(pu)
pu.cb_top_geographies(
    funding_by_country,
    "no_of_rounds",
    value_label="Number of deals",
)

# %%
importlib.reload(pu)
pu.cb_top_geographies(
    funding_by_city,
    value_column="no_of_rounds",
    value_label="Number of deals",
    category_column="org_city",
)

# %%
importlib.reload(pu)
pu.cb_top_geographies(
    funding_by_country,
    value_column="raised_amount_gbp",
    value_label="Raised amount (£1000s)",
)

# %%
importlib.reload(au)
funding_geo_ts = au.cb_get_timeseries_by_geo(
    cb_companies_with_funds,
    funding_df,
    geographies=["United States", "United Kingdom", "China", "Germany", "India"],
    period="year",
    min_year=2010,
    max_year=2021,
)

# %%
importlib.reload(pu)
pu.time_series_by_category(
    funding_geo_ts,
    value_column="no_of_rounds",
    #     value_label = 'Raised amount (£1000s)'
)

# %%
importlib.reload(pu)
pu.time_series_by_category(
    funding_geo_ts,
    value_column="raised_amount_gbp_total",
    #     value_label = 'Raised amount (£1000s)'
)

# %%
importlib.reload(pu)
funding_by_city = au.cb_funding_by_geo(
    cb_orgs.query('country == "United Kingdom"'), funding_df, "org_city"
)
pu.cb_top_geographies(
    funding_by_city,
    value_column="no_of_rounds",
    value_label="Number of deals",
    category_column="org_city",
)

# %%
importlib.reload(au)

pu.cb_top_geographies(
    au.cb_companies_by_geo(cb_companies),
    value_column="no_of_companies",
    value_label="Number of companies",
    category_column="country",
)

# %%
importlib.reload(au)

pu.cb_top_geographies(
    au.cb_companies_by_geo(
        cb_companies.query('country == "United Kingdom"'), geo_entity="city"
    ),
    value_column="no_of_companies",
    value_label="Number of companies",
    category_column="city",
)


# %% [markdown]
# ## Digital technologies
# - Select companies in industries ("digital")
# - Get the number of companies founded by year (by industry)
# - Get the number of deals by year (by industry)
# - Long term trends (5 year trend)
# - Short term trends (2020 vs 2021)

# %%
# Which companies are in digital
importlib.reload(utils)
digital = utils.get_digital_companies(cb_companies_with_funds, CB)

# %%
importlib.reload(utils)
utils.digital_proportion(cb_companies_with_funds, digital)

# %%
importlib.reload(utils)
utils.digital_proportion(cb_companies_with_funds, digital, since=2011)

# %%
importlib.reload(au)
au.cb_top_industries(digital, CB).head(15)

# %%
importlib.reload(utils)
digital_fraction_ts = utils.digital_proportion_ts(
    cb_companies_with_funds, digital, 1998, 2021
)

# %%
importlib.reload(pu)
pu.time_series(digital_fraction_ts, y_column="digital_fraction")

# %%
importlib.reload(au)
(
    rounds_by_industry_ts,
    companies_by_industry_ts,
    investment_by_industry_ts,
) = au.investments_by_industry_ts(
    digital.drop("industry", axis=1),
    utils.DIGITAL_INDUSTRIES,
    #     ['software', 'apps'],
    CB,
    "no_of_rounds",
    2011,
    2021,
)


# %%
importlib.reload(au)
(
    rounds_by_group_ts,
    companies_by_group_ts,
    investment_by_group_ts,
) = au.investments_by_industry_ts(
    digital.drop("industry", axis=1),
    utils.DIGITAL_INDUSTRY_GROUPS,
    #     ['software', 'apps'],
    CB,
    "no_of_rounds",
    2011,
    2021,
    True,
)


# %%
importlib.reload(au)
rounds_by_industry_ts_ma = au.ts_moving_average(rounds_by_industry_ts)

# %%
# pu.time_series(companies_by_industry_ts.reset_index(), y_column="advice")

# %%
cat = "data and analytics"
cat = "apps"
cat = "hardware"
pu.time_series(companies_by_group_ts.reset_index(), y_column=cat)

# %%
pu.time_series(investment_by_group_ts.reset_index(), y_column=cat)

# %%
pu.time_series(rounds_by_group_ts.reset_index(), y_column=cat)

# %%
# CB.industry_to_group['computer']

# %%
# CB.group_to_industries['artificial intelligence']

# %%
importlib.reload(au)
au.compare_years(investment_by_group_ts).query("reference_year!=0").sort_values(
    "growth", ascending=False
)


# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(rounds_by_group_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average number of deals")

# %%
# magnitude_growth.sort_values('growth', ascending=False).head(50)

# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(companies_by_group_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average number of new companies")

# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(investment_by_group_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average investment amount")

# %% [markdown]
# ## What's in the parenting / child education companies?
#
# - Embeddings
# - Clustering
# - Review of the clusters

# %%
# Make the dataset
cb_orgs_parenting = (
    CB.get_companies_in_industries(utils.PARENT_INDUSTRIES)
    .pipe(select_by_role, "company")
    .pipe(au.get_companies_with_funds)
)

# %%
all_ids = cb_orgs_parenting.id.to_list() + cb_companies_with_funds.id.to_list()

# %%
cb_all_orgs = CB.cb_organisations.query("id in @all_ids")

# %%
len(cb_all_orgs)

# %%
from innovation_sweet_spots.utils import text_processing_utils as tpu
from innovation_sweet_spots import PROJECT_DIR
import umap
import hdbscan
import altair as alt

# %%
import innovation_sweet_spots.utils.embeddings_utils as eu
from innovation_sweet_spots.utils.embeddings_utils import QueryEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# %%
company_docs = tpu.create_documents_from_dataframe(
    cb_all_orgs, ["short_description", "long_description"]
)

# %%
vector_filename = "vectors_2022_03_02"
embedding_model = EMBEDDING_MODEL
PARENTING_DIR = PROJECT_DIR / "outputs/finals/parenting"
EMBEDINGS_DIR = PARENTING_DIR / "embeddings"

# %%
v = eu.Vectors(
    filename=vector_filename, model_name=EMBEDDING_MODEL, folder=EMBEDINGS_DIR
)
v.vectors.shape

# %%
len(v.get_missing_ids(cb_all_orgs.id.to_list()))

# %%
v.generate_new_vectors(
    new_document_ids=cb_all_orgs.id.to_list(), texts=company_docs, force_update=False
)

# %%
# v.save_vectors("vectors_2022_04_26", EMBEDINGS_DIR)

# %%
ids_to_cluster = cb_orgs_parenting.id.to_list()
vectors = v.select_vectors(ids_to_cluster)

# %%
UMAP_PARAMS = {
    "n_neighbors": 5,
    "min_dist": 0.01,
}
# Create a 2D embedding
reducer = umap.UMAP(n_components=2, random_state=21, **UMAP_PARAMS)
embedding = reducer.fit_transform(vectors)
# Create another low-dim embedding for clustering
reducer_clustering = umap.UMAP(n_components=25, random_state=1, **UMAP_PARAMS)
embedding_clustering = reducer_clustering.fit_transform(vectors)


# %%
# Clustering with hdbscan
np.random.seed(11)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    cluster_selection_method="leaf",
    prediction_data=True,
)
clusterer.fit(embedding_clustering)

soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
soft_cluster = [np.argmax(x) for x in soft_clusters]

# %%
# Prepare dataframe for visualisation
df = (
    cb_all_orgs.set_index("id")
    .loc[ids_to_cluster, :]
    .reset_index()[["id", "name", "short_description", "long_description", "country"]]
    .copy()
)
df = df.merge(CB.get_company_industries(df, return_lists=True), on=["id", "name"])
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]
df["cluster"] = [str(x) for x in clusterer.labels_]
df["soft_cluster"] = [str(x) for x in soft_cluster]

# Visualise using altair (NB: -1=points haven't been assigned to a cluster)
fig = (
    alt.Chart(df, width=500, height=500)
    .mark_circle(size=60)
    .encode(
        x="x",
        y="y",
        tooltip=[
            "soft_cluster",
            "cluster",
            "name",
            "short_description",
            "long_description",
            "country",
            "industry",
        ],
        color="soft_cluster",
    )
).interactive()

# fig

# %%
fig

# %%
from innovation_sweet_spots.utils import cluster_analysis_utils

importlib.reload(cluster_analysis_utils)

# %%
cluster_labels = []
cluster_texts = []
for c in df.cluster.unique():
    ct = [corpus_full[id_] for id_ in df.query("cluster == @c").id.to_list()]
    cluster_labels += [c] * len(ct)
    cluster_texts += ct

# %%
cluster_labels = []
cluster_texts = []
for c in df.soft_cluster.unique():
    ct = [corpus_full[id_] for id_ in df.query("soft_cluster == @c").id.to_list()]
    cluster_labels += [c] * len(ct)
    cluster_texts += ct

# %%
len(cluster_texts), len(cluster_labels)

# %%
cluster_keywords = cluster_analysis_utils.cluster_keywords(
    cluster_texts,
    cluster_labels,
    11,
    tokenizer=(lambda x: x),
    max_df=0.9,
    min_df=0.2,
)

# %%
for key in sorted(cluster_keywords.keys()):
    print(key, cluster_keywords[key])

# %%
df["cluster_description"] = df["soft_cluster"].apply(lambda x: cluster_keywords[x])

# %%
soft_cluster_prob = [
    soft_clusters[i, int(c)] for i, c in enumerate(df["soft_cluster"].to_list())
]
df["soft_cluster_prob"] = soft_cluster_prob

# %%
df_ = df.merge(CB.cb_organisations[["id", "cb_url", "homepage_url"]], how="left")

# %%
df_.to_csv(OUTPUTS_DIR / "cb_companies_parenting_v2022_04_27.csv", index=False)

# %%
# corpus_all_ids = np.array(Query.document_ids)
# ids_to_cluster = np.array(ids_to_cluster)
# # corpus_ids = [np.where(doc_id == corpus_all_ids)[0][0] for doc_id in ids_to_cluster]

# %%
# ids_to_cluster

# %%
# for c in corpus_all_ids np.where(c == ids_to_cluster)
