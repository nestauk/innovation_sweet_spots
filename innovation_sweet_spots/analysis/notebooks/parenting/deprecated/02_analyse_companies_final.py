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
import altair as alt
import numpy as np

# %%
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd

# %%
OUTPUTS_DIR = PROJECT_DIR / "outputs/finals/parenting/cb_companies"

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

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=PROJECT_DIR / "outputs/parenting/figures/")

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
reviewed_df_parenting


# %%
def map_child_comments_to_user(txt):
    cats = {
        "Sports": "Children",
        "General": "Children",
        "Learning": "Children",
        "Numerical / coding": "Children",
        "Parental support": "Parents",
        "Child care": "Parents",
        "Numerical / stem": "Children",
        "Learning / Play": "Children",
        "Parental support / Activities": "Parents",
        "Education management": "Parents",
        "Older kids": "Children",
        "Tech": "Children",
        "Child Care ": "Parents",
        "Sharing memories": "Parents",
        "Sharing memories ": "Parents",
        "Parental suport": "Parents",
        "Literacy": "Children",
    }
    if type(txt) is str:
        return cats[txt]
    else:
        return "Children"


def map_parent_comments_to_user(txt):
    cats = {
        "Helping babies to sleep": "Parents",
        "Literacy": "Children",
        "Media": "Children",
        "Reading stories": "Children",
        "Reading stories / Parental support": "Children",
        "Sharing memories": "Parents",
        "Play": "Children",
        "Activities": "Parents",
        "child care": "Parents",
        "Community": "Parents",
        "Educational management": "Parents",
        "Learning": "Children",
        "Parental support": "Parents",
        "Share memories": "Parents",
        "Stories": "Children",
        "Activities / outdoors / Play": "Parents",
        "Child care": "Parents",
        "Education management": "Parents",
        "Educational": "Children",
        "Educational / Education management": "Parents",
        "Educational / platform": "Parents",
        "Educational / special needs": "Children",
        "Learning play": "Children",
        "Parental support / activities": "Parents",
        "Parental support / community": "Parents",
        "Pregancy / health": "Parents",
        "Pregnancy": "Parents",
        "Robots": "Children",
        "Toys / Play": "Children",
        "Finance": "Parents",
        "Kids products / retail": "Parents",
        "Fertility": "Parents",
        "Adoption": "Parents",
        "Educational / health": "Parents",
        "Learning / special needs": "Parents",
        "Learning play / Outdoors": "Children",
        "Parental support ": "Parents",
        "Parental support / co-parenting": "Parents",
        "Parental support / Community": "Parents",
        "Play, activities": "Children",
        "Activities / Play": "Children",
        "Parental support  / Community": "Parents",
        "Parental support / Activities": "Parents",
        "Play / games": "Children",
        "Clothes / Kids products": "Parents",
        "Helping babies sleep": "Parents",
        "Parental support / health": "Parents",
        "Robots / hardware": "Children",
        "Robots / Tracking babies rhythms": "Children",
        "Tracking babies rhythms": "Parents",
        "Parental support / Child care": "Parents",
        "Parental support / Communities": "Parents",
    }
    if type(txt) is str:
        return cats[txt]
    else:
        return "Parents"


# %%
companies_parenting_df = reviewed_df_parenting.query('relevancy == "relevant"').assign(
    user=lambda df: df.comment.apply(map_parent_comments_to_user)
)
companies_parenting_df["interesting"] = (
    companies_parenting_df["Unnamed: 16"].str.lower().str.contains("interesting")
)

companies_child_ed_df = reviewed_df_child_ed.query(
    'relevancy == "relevant" or comment == "potentially relevant"'
).assign(user=lambda df: df.comment.apply(map_child_comments_to_user))
companies_child_ed_df["interesting"] = (
    companies_child_ed_df.interesting.isnull() == False
)

# %%
id_to_user = pd.concat(
    [
        companies_parenting_df[["id", "user", "interesting"]],
        companies_child_ed_df[["id", "user", "interesting"]],
    ],
    ignore_index=False,
).fillna({"interesting": False})

# %%
companies_ids = set(companies_parenting_df.id.to_list()).union(
    set(companies_child_ed_df.id.to_list())
)
custom_ids = set(
    [
        "95487399-812c-d898-a435-c9494023cbbc",
        "58a73d1f-036b-4f21-875d-dfa3f3ef93be",
    ]
)
companies_ids = companies_ids.union(custom_ids)

# %%
len(companies_ids)

# %%
# CB.cb_organisations.query("id == '58a73d1f-036b-4f21-875d-dfa3f3ef93be'")[check_columns]

# %%
# CB.cb_organisations[CB.cb_organisations.name.str.contains("Maple") & (CB.cb_organisations.name.isnull()==False)]
# CB.cb_organisations[(CB.cb_organisations.cb_url=="https://www.crunchbase.com/organization/maple-93be") & (CB.cb_organisations.name.isnull()==False)]

# %% [markdown]
# # Analyse parenting companies

# %% [markdown]
# ## Selection

# %%
# importlib.reload(utils)
# cb_orgs = CB.get_companies_in_industries(utils.PARENT_INDUSTRIES)

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
funding_df = (
    CB.get_funding_rounds(cb_companies_with_funds)
    #     .query("investment_type in @utils.LATE_STAGE_DEALS")
    .query("investment_type in @utils.EARLY_STAGE_DEALS")
)

# %%
len(funding_df)

# %% [markdown]
# ## Export tables

# %%
CB.get_funding_round_investors(funding_df).info()

# %%
import numpy as np

# %%
investors = (
    CB.get_funding_round_investors(funding_df)
    .groupby(["investor_name"])
    .agg(
        raised_amount_gbp=("raised_amount_gbp", "sum"),
        company_names=("name", lambda x: sorted(list(np.unique(list(x))))),
    )
    .sort_values("raised_amount_gbp", ascending=False)
    .reset_index()
    .merge(
        CB.cb_investors[
            [
                "id",
                "name",
                "country_code",
                "city",
                "facebook_url",
                "linkedin_url",
                "twitter_url",
                "roles",
            ]
        ],
        how="left",
        left_on="investor_name",
        right_on="name",
    )
    .merge(
        CB.cb_organisations[["id", "homepage_url"]],
        how="left",
    )
)

# %%
columns = [
    "investor_name",
    "country_code",
    "city",
    "raised_amount_gbp",
    "company_names",
    "homepage_url",
    "facebook_url",
    "linkedin_url",
    "twitter_url",
]

# %%
for p in columns:
    print(p)

# %%
# (
#     investors[columns].to_csv(
#         PROJECT_DIR / "outputs/finals/parenting/investors_list_all.csv", index=False
#     )
# )

# %%
# CB.cb_organisations.head(3).info()

# %%
df = (
    cb_companies.merge(id_to_user).merge(
        funding_df.groupby("org_id")
        .sum()
        .reset_index()[["org_id", "raised_amount_gbp"]],
        left_on="id",
        right_on="org_id",
        how="left",
    )
)[
    [
        "id",
        "name",
        "cb_url",
        "country_code",
        "city",
        "homepage_url",
        "short_description",
        "long_description",
        "total_funding_usd",
        "raised_amount_gbp",
        "last_funding_on",
        "facebook_url",
        "linkedin_url",
        "twitter_url",
        "user",
        "interesting",
    ]
]
df.to_csv(PROJECT_DIR / "outputs/finals/parenting/company_list.csv", index=False)

# %%
for p in df.columns.to_list():
    print(p)

# %% [markdown]
# ## Generate graphs
# - Select specific deal types (earlier stage)
# - Check global
# - Check UK vs global
# - Check a baseline growth rate
# - Report the fraction of digital
# - Which digital categories are strong, and emerging?

# %%
funding_df = (
    CB.get_funding_rounds(cb_companies_with_funds)
    #     .query("investment_type in @utils.LATE_STAGE_DEALS")
    .query("investment_type in @utils.EARLY_STAGE_DEALS")
)

# %%
funding_ts = au.cb_get_all_timeseries(
    cb_companies_with_funds, funding_df, "year", 2010, 2021
)

# %%
funding_ts.head(3)

# %%
importlib.reload(pu)

# %%
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period < 2022")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(f"{values_label}:Q", scale=alt.Scale(domain=[0, 1200])),
        tooltip=tooltip,
    )
)
fig_final = pu.configure_plots(fig)
fig_final

# %%
AltairSaver.save(
    fig_final, f"parenting_tech_Total_investment", filetypes=["png", "html"]
)

# %%
au.percentage_change(
    data.query("`Year`==2011")[values_label].iloc[0],
    data.query("`Year`==2021")[values_label].iloc[0],
)

# %%
data.query("`Year`==2021")[values_label].iloc[0] / data.query("`Year`==2011")[
    values_label
].iloc[0]

# %%
au.percentage_change(
    data.query("`Year`==2020")[values_label].iloc[0],
    data.query("`Year`==2021")[values_label].iloc[0],
)

# %%
au.smoothed_growth(data.assign(year=lambda df: df["Year"].dt.year), 2017, 2021)

# %% [markdown]
# ### Baseline

# %%
funding_ts = au.cb_get_all_timeseries(
    cb_companies_with_funds, funding_df, "year", 2010, 2021
)

# %% [markdown]
# ### UK vs the world

# %%
cb_companies_with_funds.groupby("country", as_index=False).agg(
    counts=("id", "count")
).sort_values("counts", ascending=False).head(10)

# %%
country_investments = (
    funding_df
    # Last five years
    .query("year >= 2017")
    .merge(cb_companies_with_funds[["id", "country"]], left_on="org_id", right_on="id")
    .groupby("country")
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"), no_of_deals=("id", "count"))
    .assign(percentage=lambda df: df.raised_amount_gbp / df.raised_amount_gbp.sum())
    .sort_values("raised_amount_gbp", ascending=False)
    .reset_index()
    .head(10)
)
country_investments

# %%
labels_label = "Country"
values_label = "Investment (million GBP)"
tooltip = [labels_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    country_investments.assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    #     .query("time_period < 2022")
    .rename(
        columns={
            "country": labels_label,
            "raised_amount_gbp": values_label,
        }
    )
)
data

# %%
fig = (
    alt.Chart(
        data,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(
            f"{values_label}:Q",
            #             "no_of_deals:Q"
            scale=alt.Scale(domain=[0, 1500]),
        ),
        alt.Y(
            f"{labels_label}:N",
            sort=data[labels_label].to_list()
            #             sort="-x"
        ),
        tooltip=tooltip,
    )
)
fig_final = pu.configure_plots(fig)
fig_final

# %%
AltairSaver.save(
    fig_final, f"parenting_tech_Country_investment", filetypes=["png", "html"]
)

# %% [markdown]
# ### Rest of the graphs

# %%
# pu.cb_investments_barplot(
#     funding_ts, y_column="no_of_rounds", y_label="Number of deals", x_label="Year"
# )

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
countries = ["United States", "United Kingdom", "China", "India"]

# %%
importlib.reload(au)
funding_geo_ts = au.cb_get_timeseries_by_geo(
    cb_companies_with_funds,
    funding_df,
    geographies=countries,
    period="year",
    min_year=2010,
    max_year=2021,
).query("time_period < 2022")

# %%
importlib.reload(pu)
pu.time_series_by_category(
    funding_geo_ts,
    value_column="raised_amount_gbp_total",
)

# %%
# Longer term growth, 2017 -> 2021
dfs = []
for country in countries:
    dfs.append(
        au.estimate_magnitude_growth(
            (
                funding_geo_ts.query("geography == @country")
                .assign(year=lambda df: pu.convert_time_period(df["time_period"], "Y"))
                .drop(["time_period", "geography"], axis=1)
            ),
            2017,
            2021,
        ).assign(country=country)
    )
dfs = pd.concat(dfs, ignore_index=True)

# %%
dfs.sort_values(["trend", "country"])

# %%
# Shorter term growth, 2020 -> 2021
dfs = []
for country in countries:
    data = funding_geo_ts.query("geography == @country").assign(
        year=lambda df: pu.convert_time_period(df["time_period"], "Y")
    )
    dfs.append(
        pd.DataFrame(
            data={
                "growth": (
                    [
                        au.percentage_change(
                            data.query("`year`==2020")["raised_amount_gbp_total"].iloc[
                                0
                            ],
                            data.query("`year`==2021")["raised_amount_gbp_total"].iloc[
                                0
                            ],
                        )
                    ]
                ),
                "country": [country],
            }
        )
    )
dfs = pd.concat(dfs, ignore_index=True)

# %%
dfs

# %% [markdown]
# ## Baseline

# %%
funding_ts = au.cb_get_all_timeseries(
    CB.cb_organisations.pipe(select_by_role, "company"),
    cb_all_funds,
    "year",
    2010,
    2021,
)

# %%
len(CB.cb_funding_rounds)

# %%
CB.cb_funding_rounds[
    CB.cb_funding_rounds.announced_on.apply(lambda x: type(x) == str)
].announced_on

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu

importlib.reload(wu)
CB = wu.CrunchbaseWrangler()

# %%
cb_all_funding_rounds = CB.get_funding_rounds(CB.cb_organisations)

# %%
df_trend = au.cb_investments_per_period(
    (cb_all_funding_rounds.query("investment_type in @utils.EARLY_STAGE_DEALS").copy()),
    period="Y",
    min_year=2010,
    max_year=2021,
).assign(year=lambda df: df.time_period.dt.year)

# %%
df_trend

# %%
4.208886e08 / 3.311076e07

# %%
au.smoothed_growth(df_trend.drop("time_period", axis=1), 2017, 2021)

# %%
au.percentage_change(
    df_trend.query("`year`==2011")["raised_amount_gbp_total"].iloc[0],
    df_trend.query("`year`==2021")["raised_amount_gbp_total"].iloc[0],
)

# %%
(142.4 - 115.4) / 142.4

# %%
au.percentage_change(
    df_trend.query("`year`==2020")["raised_amount_gbp_total"].iloc[0],
    df_trend.query("`year`==2021")["raised_amount_gbp_total"].iloc[0],
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
utils.digital_proportion(cb_companies, digital)

# %%
importlib.reload(utils)
utils.digital_proportion(cb_companies, digital, since=2011)

# %%
digital_ids = digital.id.to_list()
# cb_companies.query("id in @digital_ids").total_funding_usd.sum()

# %%
importlib.reload(au)
top_industries = au.cb_top_industries(digital, CB)

# %%
# top_industries.query("industry in @utils.DIGITAL_INDUSTRIES").head(50)

# %%
importlib.reload(utils)
digital_fraction_ts = utils.digital_proportion_ts(cb_companies, digital, 1998, 2021)

# %%
# digital_fraction_ts

# %%
# importlib.reload(pu)
# pu.cb_investments_barplot(
#     digital_fraction_ts,
#     y_column="digital_fraction",
#     x_label="Time period",
# )

# %%
# importlib.reload(pu)
# pu.time_series(digital_fraction_ts, y_column="digital_fraction")

# %%
digital_fraction_ts.head(5)

# %%
horizontal_label = "Year"
values_label = "Companies in digital"
tooltip = [horizontal_label, alt.Tooltip(values_label, format="%")]

data = (
    digital_fraction_ts.query("time_period < 2022")
    .assign(year=lambda df: pu.convert_time_period(df["time_period"], "Y"))
    .rename(
        columns={
            "year": horizontal_label,
            "digital_fraction": values_label,
        }
    )
)
# data

# %%
label_expression = (
    " : ".join([f"datum.label == {y} ? '{y}'" for y in list(range(2000, 2021, 5))])
    + " : null"
)
label_expression

# %%
fig = (
    alt.Chart(
        data,
        width=350,
    )
    .mark_area(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X(
            f"{'time_period'}:T",
            axis=alt.Axis(
                grid=False,
                labelAlign="center",
                tickCount=20,
                labelAngle=-0,
                labelExpr=label_expression,
                title="Year",
            ),
        ),
        y=alt.Y(
            f"{values_label}:Q",
            axis=alt.Axis(
                grid=False,
                format="%",
            ),
            scale=alt.Scale(domain=(0, 1)),
        ),
    )
    .configure_axis(
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)
fig

# %% [markdown]
# ### Trends for specific digital categories

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
    CB,
    "no_of_rounds",
    2011,
    2021,
    False,
    utils.EARLY_STAGE_DEALS,
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
    utils.EARLY_STAGE_DEALS,
)


# %%
n_rounds_for_groups = pd.DataFrame(
    rounds_by_group_ts.reset_index()
    .query("time_period >= 2017 and time_period < 2022")
    .set_index("time_period")
    .sum(),
    columns=["counts"],
)

n_rounds_for_industries = pd.DataFrame(
    rounds_by_industry_ts.reset_index()
    .query("time_period >= 2017 and time_period < 2022")
    .set_index("time_period")
    .sum(),
    columns=["counts"],
)

# %%
comp_industries = CB.get_company_industries(digital)

# %%
# https://altair-viz.github.io/gallery/area_chart_gradient.html
importlib.reload(pu)
importlib.reload(au)
magnitude_growth = au.ts_magnitude_growth(investment_by_industry_ts, 2017, 2021)
pu.magnitude_growth(magnitude_growth, "Average investment amount")

# %%
df = (
    magnitude_growth.query("magnitude!=0")
    .query("growth != inf")
    .sort_values("growth", ascending=False)
)
df["counts"] = n_rounds_for_industries["counts"]
df["company_counts"] = comp_industries.groupby("industry").agg(
    company_counts=("id", "count")
)["company_counts"]

# %%
df

# %%
labels_to_show = {
    "machine learning": "machine learning",
    "ebooks": "ebooks",
    "online portals": "online portals",
    "content": "content",
    "edtech": "edtech",
    "saas": "saas",
    "e-learning": "e-learning",
    "edtech": "edtech",
    "android": "android",
    "apps": "apps",
    "saas": "saas",
    "ios": "ios",
    "augmented reality": "AR",
}


def produce_labels(label):
    if label in labels_to_show:
        return labels_to_show[label]
    else:
        return ""


df_filtered = (
    df.query("company_counts>=5 and counts>=10")
    .drop(["edtech", "e-learning"])
    .reset_index()
    .rename(columns={"index": "digital_technology"})
    .assign(text_label=lambda df: df.digital_technology.apply(produce_labels))
    .assign(
        Increase=lambda df: df.growth.apply(
            lambda x: "positive" if x >= 0 else "negative"
        )
    )
)

# %%
df_filtered

# %%
# # https://altair-viz.github.io/gallery/area_chart_gradient.html
# importlib.reload(pu)
# importlib.reload(au)
# magnitude_growth = au.ts_magnitude_growth(investment_by_industry_ts, 2017, 2021)
# pu.magnitude_growth(df_filtered, "Average investment amount")

# %%
fig = (
    alt.Chart(
        (
            df_filtered.assign(
                growth=lambda df: df.growth / 100,
                magnitude=lambda df: df.magnitude / 1000,
            )
        ),
        width=350,
        height=300,
    )
    .mark_circle(size=50, color=pu.NESTA_COLOURS[0], clip=True, opacity=0.6)
    .encode(
        x=alt.X(
            "magnitude:Q",
            axis=alt.Axis(
                title=f"Average yearly investment (million GBP)", labelAlign="center"
            ),
            scale=alt.Scale(domain=(0, 120))
            # scale=alt.Scale(type="linear"),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(format="%", title="Growth"),
            scale=alt.Scale(domain=(-1, 40))
            # axis=alt.Axis(
            #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            # ),
            # scale=alt.Scale(domain=(-.100, .300)),
        ),
        # size="cluster_size:Q",
        #         color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=[
            alt.Tooltip("digital_technology:N", title="Digital technology"),
            alt.Tooltip(
                "magnitude:Q",
                format=",",
                title="Average yearly investment (million GBP)",
            ),
            alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        ],
    )
)

text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
    text="text_label:N"
)

fig_final = (
    (fig + text)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="white",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
    .interactive()
)
fig_final

# %%
fig = (
    alt.Chart(
        (
            df_filtered.assign(
                growth=lambda df: df.growth / 100,
                magnitude=lambda df: df.magnitude / 1000,
            )
        ),
        width=300,
        height=450,
    )
    .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1)
    .encode(
        x=alt.X(
            "growth:Q",
            axis=alt.Axis(
                format="%",
                title="Growth",
                #                 tickCount=25,
                labelAlign="center",
                labelExpr="datum.value < -1 ? null : datum.label",
            ),
            #             axis=alt.Axis(title=f"Average yearly investment (million GBP)", labelAlign='center'),
            scale=alt.Scale(domain=(-1, 37)),
            # scale=alt.Scale(type="linear"),
        ),
        y=alt.Y(
            "digital_technology:N",
            sort="-x",
            axis=alt.Axis(title="Digital category"),
            #             scale=alt.Scale(domain=(-1,40))
            # axis=alt.Axis(
            #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            # ),
            # scale=alt.Scale(domain=(-.100, .300)),
        ),
        size=alt.Size(
            "magnitude",
            title="Yearly investment (million GBP)",
            legend=alt.Legend(orient="top"),
        ),
        color=alt.Color(
            "Increase",
            sort=["positive", "negative"],
            legend=None,
            scale=alt.Scale(
                domain=["positive", "negative"],
                range=[pu.NESTA_COLOURS[0], pu.NESTA_COLOURS[4]],
            ),
        ),
        # size="cluster_size:Q",
        #         color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=[
            alt.Tooltip("digital_technology:N", title="Digital technology"),
            alt.Tooltip(
                "magnitude:Q",
                format=",.3f",
                title="Average yearly investment (million GBP)",
            ),
            alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        ],
    )
)

# text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
#     text='text_label:N'
# )

fig_final = (
    (fig)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        labelFontSize=pu.FONTSIZE_NORMAL - 1,
        titleFontSize=pu.FONTSIZE_NORMAL - 1,
    )
    .configure_view(strokeWidth=0)
    #     .interactive()
)
fig_final

# %%
AltairSaver.save(fig_final, f"parenting_tech_Digital", filetypes=["png", "html"])

# %%

# %%
industry_name = "internet of things"
industry_name = "gaming"
pd.set_option("max_colwidth", 200)
ids = comp_industries[comp_industries.industry == industry_name].id.to_list()
digital.query("id in @ids")[
    check_columns + ["total_funding_usd", "last_funding_on", "country", "homepage_url"]
].astype({"total_funding_usd": float}).sort_values("total_funding_usd", ascending=False)

# %%

# %% [raw]
# (
#     magnitude_growth
#     .query("magnitude!=0")
#     .query("growth != inf")
#     .sort_values('growth', ascending=False)
# ).head(20)

# %%

# %%
importlib.reload(au)
rounds_by_industry_ts_ma = au.ts_moving_average(rounds_by_industry_ts)

# %%
cat = "gaming"
# cat = "apps"
pu.time_series(companies_by_industry_ts.reset_index(), y_column=cat)

# %%
pu.time_series(investment_by_industry_ts.reset_index(), y_column=cat)

# %%
pu.time_series(rounds_by_industry_ts.reset_index(), y_column=cat)

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
magnitude_growth_industry = au.ts_magnitude_growth(
    investment_by_industry_ts, 2017, 2021
)
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

# %% [markdown]
# ## What's in 'family' companies

# %%
# Make the dataset
cb_orgs_family = (
    CB.get_companies_in_industries(["family"])
    .pipe(select_by_role, "company")
    .pipe(au.get_companies_with_funds)
)

# %% [markdown]
# ###

# %%
