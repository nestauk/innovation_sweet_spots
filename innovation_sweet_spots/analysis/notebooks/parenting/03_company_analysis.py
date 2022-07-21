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
# # Venture capital trend analysis
#
# This notebook produces graphs for the venture capital (VC) trend analysis for early years education and parenting companies.
#

# %%
from innovation_sweet_spots.analysis.notebooks.parenting import utils
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
import innovation_sweet_spots.utils.altair_save_utils as alt_save
import innovation_sweet_spots.analysis.analysis_utils as au
import innovation_sweet_spots.utils.plotting_utils as pu
import pandas as pd
import altair as alt

# Import reviewed companies
PARENTING_DIR = PROJECT_DIR / "outputs/parenting/cb_companies"
FIGURES_DIR = PROJECT_DIR / "outputs/parenting/figures/"
TABLES_DIR = FIGURES_DIR / "tables"

# %%
CB = CrunchbaseWrangler()
AltairSaver = alt_save.AltairSaver(path=FIGURES_DIR)

# %% [markdown]
# # Fetching data

# %%
# Fetch reviewed companies
reviewed_companies = pd.read_csv(PARENTING_DIR / "cb_companies_ids_reviewed.csv")
# Company identifiers
companies_ids = set(reviewed_companies.id.to_list())
# Double check that ony companies with funds have been selected
cb_companies_with_funds = utils.select_companies_with_funds(companies_ids, CB)
# Funding data
funding_df = CB.get_funding_rounds(cb_companies_with_funds).query(
    "investment_type in @utils.EARLY_STAGE_DEALS"
)
# Funding time series
funding_ts = au.cb_get_all_timeseries(
    cb_companies_with_funds, funding_df, "year", 2009, 2021
)

# %%
utils.EARLY_STAGE_DEALS

# %% [markdown]
# # Drawing graphs

# %% [markdown]
# ## Total global investment

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
    .assign(
        **{
            horizontal_label: lambda df: pu.convert_time_period(
                df[horizontal_label], "Y"
            )
        }
    )
)[[horizontal_label, values_label]]

fig = (
    alt.Chart(
        data.query("Year > 2009"),
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
figure_name = f"parenting_tech_Total_investment"
AltairSaver.save(fig_final, figure_name, filetypes=["png", "html", "svg"])
utils.save_data_table(data, figure_name, TABLES_DIR)

# %% [markdown]
# ### Investment growth figures

# %%
au.smoothed_growth(data.rename(columns={"Year": "year"}), 2011, 2021)

# %%
au.smoothed_growth(data.rename(columns={"Year": "year"}), 2017, 2021)

# %%
# Percentage change from 2011 to 2021
au.percentage_change(
    data.query("`Year`==2011")[values_label].iloc[0],
    data.query("`Year`==2021")[values_label].iloc[0],
)

# %%
# Growth factor from 2011 to 2021
data.query("`Year`==2021")[values_label].iloc[0] / data.query("`Year`==2011")[
    values_label
].iloc[0]

# %%
# Percentage change from 2020 to 2021
au.percentage_change(
    data.query("`Year`==2020")[values_label].iloc[0],
    data.query("`Year`==2021")[values_label].iloc[0],
)

# %% [markdown]
# ### Baseline growth figures
#
# Calculate a baseline VC funding growth (all funding in Crunchbase)

# %%
# Get all funding rounds
cb_all_funding_rounds = CB.get_funding_rounds(CB.cb_organisations)
# Get time series of total investment
cb_all_rounds_ts = au.cb_investments_per_period(
    (cb_all_funding_rounds.query("investment_type in @utils.EARLY_STAGE_DEALS").copy()),
    period="Y",
    min_year=2009,
    max_year=2021,
).assign(year=lambda df: df.time_period.dt.year)


# %%
cb_all_rounds_ts

# %%
au.smoothed_growth(cb_all_rounds_ts.drop("time_period", axis=1), 2011, 2021)

# %%
au.smoothed_growth(cb_all_rounds_ts.drop("time_period", axis=1), 2017, 2021)

# %%
# Percentage change from 2011 to 2021
au.percentage_change(
    cb_all_rounds_ts.query("`year`==2011")["raised_amount_gbp_total"].iloc[0],
    cb_all_rounds_ts.query("`year`==2021")["raised_amount_gbp_total"].iloc[0],
)

# %%
# Growth factor from 2011 to 2021
cb_all_rounds_ts.query("`year`==2021")["raised_amount_gbp_total"].iloc[
    0
] / cb_all_rounds_ts.query("`year`==2011")["raised_amount_gbp_total"].iloc[0]


# %%
# Percentage change from 2020 to 2021
au.percentage_change(
    cb_all_rounds_ts.query("`year`==2020")["raised_amount_gbp_total"].iloc[0],
    cb_all_rounds_ts.query("`year`==2021")["raised_amount_gbp_total"].iloc[0],
)

# %%
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    cb_all_rounds_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period < 2022")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
    .assign(
        **{
            horizontal_label: lambda df: pu.convert_time_period(
                df[horizontal_label], "Y"
            )
        }
    )
)[[horizontal_label, values_label]]


fig = (
    alt.Chart(
        data.query("Year > 2009"),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(f"{values_label}:Q"),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final

# %%
figure_name = f"parenting_tech_Global_all_VC_investment"
AltairSaver.save(fig_final, figure_name, filetypes=["png", "html", "svg"])
utils.save_data_table(data, figure_name, TABLES_DIR)

# %% [markdown]
# The investment in our identified early years and parenting companies has grown about 15-16x comparing 2021 and 2011, whereas the baseline investment has grown about 11-12x for that same comparison.
#
# Using smoothed growth estimates (ie, average between 2009-2011 vs 2019-2021) the difference is even more stark, about 2000% vs 900% (ie, 21x vs 10x). Howevever, this would be harder to compare with the figures from other VC sector reports.
#
# Looking at the other report, for the global total VC figures Dealroom reports about 11x increase on their platform, Crunchbase [seems to be around 10x mark](https://news.crunchbase.com/business/global-vc-funding-unicorns-2021-monthly-recap/) (since 2012), whereas The Economist appears to have a lower figure (looking at [the graphs](https://www.economist.com/finance-and-economics/2021/11/23/the-bright-new-age-of-venture-capital/21806438), perhaps around 8x?).
#
# Given that these are inevitably approximate estimates, I think we can roughly say the figures to be about 15x (conservative estimate) for parenting and early years sector vs 10x growth for global VC in the last decade.
#
#

# %% [markdown]
# ## Country investment figures

# %%
labels_label = "Country"
values_label = "Investment (million GBP)"
tooltip = [labels_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_df
    # Last five years
    .query("year >= 2017 and year < 2022")
    .merge(cb_companies_with_funds[["id", "country"]], left_on="org_id", right_on="id")
    .groupby("country")
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"), no_of_deals=("id", "count"))
    .assign(percentage=lambda df: df.raised_amount_gbp / df.raised_amount_gbp.sum())
    .sort_values("raised_amount_gbp", ascending=False)
    .reset_index()
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .rename(
        columns={
            "country": labels_label,
            "raised_amount_gbp": values_label,
        }
    )
    .head(10)
)


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
figure_name = f"parenting_tech_Country_investment"
AltairSaver.save(fig_final, figure_name, filetypes=["png", "html", "svg"])
utils.save_data_table(data[[labels_label, values_label]], figure_name, TABLES_DIR)


# %% [markdown]
# ## Digital technologies

# %%
# Which companies are in digital
digital = utils.get_digital_companies(cb_companies_with_funds, CB)

# Get investment time series by company industry categories
# (takes a few minutes to complete)
(
    rounds_by_industry_ts,
    companies_by_industry_ts,
    investment_by_industry_ts,
) = au.investments_by_industry_ts(
    digital.drop("industry", axis=1),
    utils.DIGITAL_INDUSTRIES,
    CB,
    2011,
    2021,
    use_industry_groups=False,
    funding_round_types=utils.EARLY_STAGE_DEALS,
)

# %%
# Number of investment rounds per industry category in the given time period
n_rounds_for_industries = pd.DataFrame(
    rounds_by_industry_ts.reset_index()
    .query("time_period >= 2017 and time_period < 2022")
    .set_index("time_period")
    .sum(),
    columns=["counts"],
)
# Magnitude vs growth plots
magnitude_growth = au.ts_magnitude_growth(investment_by_industry_ts, 2017, 2021)
# All industries in the digital industry groups
comp_industries = CB.get_company_industries(digital)
# Prepare data for the graph
data = (
    # Remove industries without investment or infite growth
    magnitude_growth.query("magnitude!=0")
    .query("growth != inf")
    .sort_values("growth", ascending=False)
    # Add company deal and company counts, and filter by them
    .assign(counts=n_rounds_for_industries["counts"])
    .assign(
        company_counts=comp_industries.groupby("industry").agg(
            company_counts=("id", "count")
        )["company_counts"]
    )
    .query("company_counts>=5 and counts>=10")
    # Drop generic categories
    .drop(["edtech", "e-learning"])
    .reset_index()
    .rename(columns={"index": "digital_technology"})
    # Create auxillary variables for plotting
    .assign(
        growth=lambda df: df.growth / 100,
        magnitude=lambda df: df.magnitude / 1000,
        Increase=lambda df: df.growth.apply(
            lambda x: "positive" if x >= 0 else "negative"
        ),
    )
)


# %%
fig = (
    alt.Chart(
        data,
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
                labelAlign="center",
                labelExpr="datum.value < -1 ? null : datum.label",
            ),
            scale=alt.Scale(domain=(-1, 37)),
        ),
        y=alt.Y(
            "digital_technology:N",
            sort="-x",
            axis=alt.Axis(title="Digital category"),
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

fig_final = (
    fig.configure_axis(
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
)

fig_final

# %%
figure_name = f"parenting_tech_Digital"
AltairSaver.save(fig_final, figure_name, filetypes=["png", "html", "svg"])
utils.save_data_table(data, figure_name, TABLES_DIR)


# %%
