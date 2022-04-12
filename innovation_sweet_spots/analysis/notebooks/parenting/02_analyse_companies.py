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

# %% [markdown]
# # Analyse 'parenting' organisations

# %%
cb_orgs_parenting = CB.get_companies_in_industries(["child care"])

# %%
cb_orgs = cb_orgs_parenting

# %%
cb_companies = cb_orgs.pipe(select_by_role, "company")

# %%
cb_companies_with_funds = au.get_companies_with_funds(cb_companies)

# %%
funding_df = CB.get_funding_rounds(cb_companies_with_funds)
funding_ts = au.cb_get_all_timeseries(
    cb_companies_with_funds, funding_df, "year", 2010, 2021
)

# %%
funding_ts

# %%
funding_df.head(3)

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
funding_df.info()

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
importlib.reload(au)
au.cb_top_industries(digital, CB)

# %%
importlib.reload(utils)
digital_fraction_ts = utils.digital_proportion_ts(cb_companies, digital, 1998, 2021)

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
# pu.time_series(rounds_by_industry_ts.reset_index(), y_column="edtech")

# %%
# CB.industry_to_group['computer']

# %%
# CB.group_to_industries['data and analytics']

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
# Add - millions or thousands
# Add - benchmarking
# Add time series for a country, and comparing countries
