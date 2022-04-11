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
# For any selection or organisations, prepares the following report:
#
# - Total investment across years
# - Number of deals across years
# - Types of investment (NB: needs simpler reporting of deal types)
# - Investment by countries (top countries)
# - Top UK cities
# - Fraction of digital among these companies
# - Investment trends by digital category
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
importlib.reload(pu)
pu.test_chart()

# %% [markdown]
# # Analyse 'parenting' organisations

# %%
cb_orgs_parenting = CB.get_companies_in_industries(["parenting"])

# %%
cb_orgs = cb_orgs_parenting

# %%
cb_companies = cb_orgs.pipe(select_by_role, "company")

# %%
cb_companies_with_funds = au.get_companies_with_funds(cb_companies)

# %%
# cb_orgs.roles.unique()

# %%
# select_by_role(cb_orgs, 'company')
