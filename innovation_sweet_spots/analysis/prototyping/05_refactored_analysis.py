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

# %%
from innovation_sweet_spots.getters.hansard import get_hansard_data
from innovation_sweet_spots.getters.guardian import search_content
from innovation_sweet_spots.getters import crunchbase, gtr

import innovation_sweet_spots.analysis.analysis_utils as iss

# %%
import importlib

importlib.reload(iss)

# %%
import pandas as pd

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %% [markdown]
# ## 1. Setup

# %%
# Import GTR data
gtr_projects = gtr.get_gtr_projects()
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# Links tables
link_gtr_funds = gtr.get_link_table("gtr_funds")
link_gtr_organisations = gtr.get_link_table("gtr_organisations")
link_gtr_topics = gtr.get_link_table("gtr_topic")

gtr_project_funds = iss.link_gtr_projects_and_funds(gtr_funds, link_gtr_funds)
funded_projects = iss.get_gtr_project_funds(gtr_projects, gtr_project_funds)
del link_gtr_funds

# %%
project_to_org = iss.link_gtr_projects_and_orgs(
    gtr_organisations, link_gtr_organisations
)

# %%
# Import Crunchbase data
cb = crunchbase.get_crunchbase_orgs_full()
cb_df = cb[-cb.id.duplicated()]
cb_df = cb_df[cb_df.country == "United Kingdom"]
cb_df = cb_df.reset_index(drop=True)
del cb
cb_investors = crunchbase.get_crunchbase_investors()
cb_investments = crunchbase.get_crunchbase_investments()
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %%
hans = get_hansard_data()
hans_docs = iss.create_documents_from_dataframe(hans, columns=["speech"])

# %%
gtr_columns = ["title", "abstractText", "techAbstractText"]
gtr_docs = iss.create_documents_from_dataframe(gtr_projects, gtr_columns)


# %%
cb_columns = ["name", "short_description", "long_description"]
cb_docs = iss.create_documents_from_dataframe(cb_df, cb_columns)

# %% [markdown]
# ### Characterise data sets

# %%
len(gtr_docs)

# %%
gtr_projects.start.min(), gtr_projects.start.max()

# %%
len(cb_df)

# %%
cb_df.roles.str.contains("investor").sum()

# %%
cb_funding_rounds[cb_funding_rounds.org_id.isin(cb_df.id.to_list())].announced_on.min()

# %%
cb_funding_rounds[cb_funding_rounds.org_id.isin(cb_df.id.to_list())].announced_on.max()

# %%
from collections import Counter
import pandas as pd

uk_funding = cb_funding_rounds[cb_funding_rounds.org_id.isin(cb_df.id.to_list())]
year = Counter((iss.convert_date_to_year(y) for y in uk_funding.announced_on.to_list()))

# %%
yearly_rounds = iss.impute_years(
    pd.DataFrame(data={"year": year.keys(), "rounds": year.values()}).sort_values(
        "year"
    ),
    min_year=1978,
    max_year=2020,
)

# %%
iss.show_time_series(yearly_rounds, y="rounds")

# %% [markdown]
# ## Define a search term
#
# Try `heat pump`, `photovoltaic`, `carbon capture`, `district heating`

# %%
search_term = "heat pump"


# %%
def plot_comp(terms=["heat pump", "fuel cell"], y="no_of_projects", show_trend=True):
    figs = []
    for search_term in terms:
        proj = iss.search_via_docs(search_term, gtr_docs, funded_projects)
        search_term_funding = iss.gtr_funding_per_year(proj, min_year=2010)
        figs.append(
            iss.show_time_series_fancier(
                search_term_funding, y=y, show_trend=show_trend
            )
        )
    fig_base = figs[0]
    for fig in figs[1:]:
        fig_base += fig
    return iss.nicer_axis(fig_base)


# %%
plot_comp()

# %% [markdown]
# ## 2. Research project analysis

# %%
proj = iss.search_via_docs(search_term, gtr_docs, funded_projects)
proj["year"] = proj["start"].apply(iss.convert_date_to_year)
search_term_funding = iss.gtr_funding_per_year(proj, min_year=2010)

# %%

# %%
pd.set_option("max_colwidth", 200)
df = (
    proj.query("year<2021").sort_values("start", ascending=False).reset_index(drop=True)
).copy()
df.amount = df.amount / 1000
df[["title", "amount"]].rename(columns={"amount": "amount (1000s)"}).head(15)

# %%
docs_with_term = iss.get_docs_with_term(search_term, gtr_docs)
sentences = iss.get_sentences_with_term(search_term, docs_with_term)

# %%
sentences[6]

# %% [markdown]
# ### Number of projects, and the funding amounts

# %%
importlib.reload(iss)

# %%
plots = ["no_of_projects", "amount_total", "amount_median"]

# %%
for plot in plots:
    fig = iss.nicer_axis(iss.show_time_series_fancier(search_term_funding, y=plot))
    alt_save.save_altair(fig, f"GTR_test_{search_term}_{plot}", driver)

# %%
plt1 = iss.show_time_series_fancier(search_term_funding, y=plots[0], show_trend=False)
iss.nicer_axis(plt1)

# %%
iss.estimate_growth_level(search_term_funding, column="no_of_projects")

# %%
plt2 = iss.show_time_series(search_term_funding, y="amount_total")
plt2

# %%
iss.estimate_growth_level(search_term_funding, column="amount_total")

# %%
plt3 = iss.show_time_series(search_term_funding, y="amount_median")
plt3

# %%
search_term_funding.iloc[-10:-5].amount_median.mean()

# %%
search_term_funding.iloc[-5:].amount_median.mean()

# %%

# %%
iss.estimate_growth_level(search_term_funding, column="amount_median")

# %%
importlib.reload(iss)

# %%
iss.show_time_series_points(df, y="amount", ymax=5000)

# %%
iss.show_time_series_points(df, y="amount", ymax=5000, clip=False)

# %%
alt_save.save_altair(
    iss.show_time_series_points(df, y="amount"),
    f"GTR_test_{search_term}_amount",
    driver,
)
alt_save.save_altair(
    iss.show_time_series_points(df, y="amount", ymax=5000, clip=False),
    f"GTR_test_{search_term}_amount_zoom",
    driver,
)

# %% [markdown]
# #### Funded organisations

# %%
project_orgs = iss.get_gtr_project_orgs(proj, project_to_org)
funded_orgs = iss.get_org_stats(project_orgs)
funded_orgs.amount_total = funded_orgs.amount_total / 1000
funded_orgs.reset_index().rename(columns={"amount_total": "total amount (1000s)"}).head(
    15
)

# %%
project_orgs[project_orgs.name.str.contains("Newcastle University")]

# %%
import innovation_sweet_spots.pipeline.network_analysis as iss_net
import innovation_sweet_spots.utils.altair_network as alt_net

# %%
org_list = (
    pd.DataFrame(project_orgs.groupby(["project_id", "name"]).count().index.to_list())
    .groupby(0)[1]
    .apply(lambda x: list(x))
    .to_list()
)

# %%
#     co_occ = (
#         project_orgs.reset_index(drop=False)
#         .melt(id_vars="index")
#         .reset_index(drop=False)
#         .groupby("index")["name"]
#         .apply(lambda x: list(x))
#     )
#     return co_occ

# %%
graph = iss_net.make_network_from_coocc(org_list, spanning=False)

# %%
import networkx as nx

importlib.reload(alt_net)
importlib.reload(iss_net)

# %%
nodes = (
    pd.DataFrame(nx.layout.spring_layout(graph, seed=1))
    .T.reset_index()
    .rename(columns={"index": "node", 0: "x", 1: "y"})
)
df = funded_orgs.reset_index().rename(columns={"name": "node"})
df["node_name"] = df["node"]
nodes = nodes.merge(df)

# %%
nodes.head()

# %%
nodes["is_university"] = nodes.node_name.str.contains(
    "University"
) | nodes.node_name.str.contains("College")
# nodes = nodes[nodes.is_university]

# %%
net_plot = alt_net.plot_altair_network(
    nodes,
    graph=graph,
    node_label="node",
    node_size="no_of_projects",
    node_size_title="number of projects",
    edge_weight_title="number of projects",
    title=f"Collaboration network",
    node_color="is_university",
    node_color_title="University",
)
net_plot.interactive()

# %%
alt_save.save_altair(net_plot, "heat_pump_network", driver)

# %% [markdown]
# ## 3. Crunchbase analysis

# %%
search_term = "artificial intelligence"

# %%
orgs_with_term = iss.search_via_docs(search_term, cb_docs, cb_df)
fund_rounds = iss.get_cb_org_funding_rounds(orgs_with_term, cb_funding_rounds)
funding_per_year = iss.get_cb_funding_per_year(fund_rounds)

# %%
funding_per_year.raised_amount_usd_total = (
    funding_per_year.raised_amount_usd_total / 1000
)

# %%
# funding_per_year.head(2)

# %%
# orgs_with_term[['name', 'total_funding', 'city', 'country']].sort_values('total_funding', ascending=False)

# %% [markdown]
# ### Funding rounds and amount over years

# %%
plt1 = iss.nicer_axis(
    iss.show_time_series_fancier(funding_per_year, y="no_of_rounds", show_trend=False)
)
# alt_save.save_altair(plt1, 'heat_pump_no_of_rounds', driver)
plt1

# %%
iss.estimate_growth_level(funding_per_year, column="no_of_rounds")

# %%
plt1 = iss.nicer_axis(
    iss.show_time_series_fancier(
        funding_per_year, y="raised_amount_usd_total", show_trend=False
    )
)
alt_save.save_altair(plt1, "heat_pump", driver)

# %%
iss.estimate_growth_level(funding_per_year, column="raised_amount_usd_total")

# %%
iss.show_time_series(iss.cb_orgs_funded_by_year(orgs_with_term), y="no_of_orgs_founded")

# %% [markdown]
# ### Companies and investors

# %%
iss.cb_orgs_with_most_funding(orgs_with_term).sort_values("founded_on", ascending=False)

# %%
fund_rounds_investors = iss.get_funding_round_investors(fund_rounds, cb_investments)
iss.investor_raised_amounts(fund_rounds_investors)

# %% [markdown]
# ## 4. Hansard analysis

# %%
search_term = "heat pump"

# %%
importlib.reload(iss)

# %%
speeches = iss.search_via_docs(search_term, hans_docs, hans)
mentions = iss.get_hansard_mentions_per_year(speeches, max_year=2021)

# %%
iss.show_time_series(mentions, y="mentions")

# %%
plt1 = iss.nicer_axis(
    iss.show_time_series_fancier(mentions, y="mentions", show_trend=False)
)
alt_save.save_altair(plt1, "heat_pump_Hansard", driver)

# %%
docs_with_term = iss.get_docs_with_term(search_term, hans_docs)
sentences = iss.get_sentences_with_term(search_term, docs_with_term)
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
# sentiment_df

# %%
for i, row in sentiment_df.iloc[0:5].iterrows():
    print(row.compound, row.sentences, end="\n\n")

# %%
for i, row in sentiment_df.sort_values("compound").iloc[-5:].iterrows():
    print(row.compound, row.sentences, end="\n\n")

# %%
sentiment_df.sort_values("pos").iloc[-2].sentences

# %% [markdown]
# ## 5. News analysis

# %%
search_term = "heat pumps"
importlib.reload(iss)
articles = search_content(search_term)

# %%
# articles[0]

# %%
df.date.to_list()

# %%
df = pd.DataFrame(
    data={
        "headline": [a["fields"]["headline"] for a in articles],
        "date": [a["webPublicationDate"] for a in articles],
    }
).head(20)
df.date = [x[0:10] for x in df.date.to_list()]
df

# %%
# select_sections = ['Environment']
# articles = [a for a in articles if a['sectionName'] in ['Environment']]

# %% [markdown]
# ### News mentions

# %%
# iss.show_time_series(iss.get_guardian_mentions_per_year(articles), y='mentions')


# %%
plt1 = iss.nicer_axis(
    iss.show_time_series_fancier(
        iss.get_guardian_mentions_per_year(articles), y="articles", show_trend=False
    )
)
plt1

# %%
alt_save.save_altair(plt1, "heat_pumps_Guardian", driver)

# %% [markdown]
# ### Assessing sentiment

# %%
yearly_sentiment = iss.news_sentiment_over_years(search_term, articles)
iss.show_time_series(yearly_sentiment, y="mean_sentiment")

# %%
plt1 = iss.nicer_axis(
    iss.show_time_series_fancier(yearly_sentiment, y="mean_sentiment", show_trend=False)
)
plt1

# %%
alt_save.save_altair(plt1, "heat_pumps_Guardian_sentiment", driver)

# %%
# year = 2015
# articles_of_year = [a for a in articles if iss.convert_date_to_year(a['webPublicationDate']) == year]
# headlines = list(iss.get_article_field(articles_of_year, 'headline'))
# for i, row in iss.get_sentence_sentiment(headlines).iterrows():
#     print(f'{row.compound}: {row.sentences}')

# %% [markdown]
# ### Extracting more granular sentiment

# %%
sentences = iss.get_guardian_sentences_with_term(search_term, articles, field="body")
sentiment_df = iss.get_sentence_sentiment(sentences)

# %%
for i, row in (
    sentiment_df.sort_values("compound", ascending=True).iloc[0:10].iterrows()
):
    print(np.round(row.compound, 2), row.sentences, end="\n\n")

# %%
for i, row in sentiment_df.sort_values("compound").iloc[-10:].iterrows():
    print(row.compound, row.sentences, end="\n\n")

# %% [markdown]
# ### Writers

# %%
iss.get_guardian_contributors(articles).head(10)

# %%
