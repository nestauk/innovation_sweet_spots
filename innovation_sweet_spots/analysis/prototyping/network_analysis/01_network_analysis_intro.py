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
# # Intro to basic network analysis
#
# **In this notebook**
# - Building a collaboration network between research organisations working on a technology
#     - Nodes = organisations, links = a common project
# - Visualising the network that allows exploration of the relationships by a non-data scientist
#
# **Some potential further directions**
#
# Using Gateway to Research data:
# 1. Interactivity
#     - Using Altair to explore the network, allowing to check organisations and what projects form the basis of the links
#     - Exploring displaying these networks in [kumu](https://kumu.io/), which is used by the ASF team
# 2. Performing simple analyses and highlighting:
#     - Which organisations are most the central ones? (i.e., most interconnected, hinting at their importance)
#     - How different types of organisations (universities, businesses, local authorities) are placed in the network, how central they are?
#     - Perhaps: Communities or connected components of organisations. Maybe a nearest-neighbour subgraph (involving all direct neighbours) is interesting to highlight
#     - Also: Can we explain the results (e.g. why a particular organisation might appear very central?)
# 3. More advanced analysis: Characterising network growth over years
#     - Visualising the new links across years
#     - Can we say something about a 'growth mechanism' ? (e.g. does it look like preferential attachment, i.e. rich get richer, or more like random growth)
#
# **Would also be very interesting to use Crunchbase data to visualise the network of investors and businesses**

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters import crunchbase, gtr
import innovation_sweet_spots.analysis.analysis_utils as iss

import innovation_sweet_spots.pipeline.network_analysis as iss_net
import innovation_sweet_spots.utils.altair_network as alt_net

import pandas as pd
import networkx as nx

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %% [markdown]
# ## Input data

# %%
INPUTS_DIR = PROJECT_DIR / "outputs/finals/"

# %%
proj_organisations = pd.read_csv(INPUTS_DIR / "ISS_projects_organisations.csv")
companies_investors = pd.read_csv(INPUTS_DIR / "ISS_companies_investors.csv")

# %% [markdown]
# # Research networks

# %%
tech_category = "Heat pumps"

# %%
# Select projects pertaining to a specific category
project_orgs = proj_organisations[proj_organisations.tech_category == tech_category]

# %%
# Check which organisations have the most project or funding
funded_orgs = iss.get_org_stats(project_orgs)
funded_orgs.amount_total = funded_orgs.amount_total / 1000
funded_orgs.reset_index().rename(columns={"amount_total": "total amount (1000s)"})

# %%
# Generate a list of organisation collaborations
org_list = (
    pd.DataFrame(project_orgs.groupby(["project_id", "name"]).count().index.to_list())
    .groupby(0)[1]
    .apply(lambda x: list(x))
    .to_list()
)
org_list[0:5]

# %%
# Build a graph
graph = iss_net.make_network_from_coocc(org_list, spanning=False)

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
# Identify academic organisations
academic_org_terms = ["university", "college", "institute", "royal academy", "research"]
gov_org_terms = ["dept", "department", "council", "comittee", "authority", "agency"]


def term_indicators(df, terms, col_name, name_col="name"):
    df = df.copy()
    x = [False] * len(df)
    for term in terms:
        x = x | df[name_col].str.lower().str.contains(term)
    df[col_name] = x
    return df


# %%
nodes = term_indicators(nodes, academic_org_terms, "is_academic", "node")

# %%
net_plot = alt_net.plot_altair_network(
    nodes,
    graph=graph,
    node_label="node",
    node_size="no_of_projects",
    node_size_title="number of projects",
    edge_weight_title="number of projects",
    title=f"Collaboration network",
    node_color="is_academic",
    node_color_title="Academic institution",
)
net_plot.interactive()

# %%
alt_save.save_altair(net_plot, "heat_pump_network", driver)

# %%
