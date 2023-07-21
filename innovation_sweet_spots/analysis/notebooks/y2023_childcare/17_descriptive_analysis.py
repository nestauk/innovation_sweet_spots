# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Descriptive analysis
#
# - Descriptive analysis (number of companies per category etc)
# - Trends per category
# - Visualise trends
#
# - Discuss improvements to the taxonomy and categorisation
# - Approach to validation

# +
from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.utils.google_sheets as gs
from innovation_sweet_spots.analysis import wrangling_utils as wu
import re
import utils
import innovation_sweet_spots.utils.plotting_utils as pu
import pandas as pd
import altair as alt
import innovation_sweet_spots.analysis.analysis_utils as au
import importlib

importlib.reload(utils)


# -

FIGURES_DIR = PROJECT_DIR / "outputs/2023_childcare/figures/"
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=FIGURES_DIR)
TABLES_DIR = FIGURES_DIR / "tables"
filetypes = ["html"]


# ## Load the data

CB = wu.CrunchbaseWrangler()

childcare_categories_df = pd.read_csv(
    PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_04_18.csv"
).drop_duplicates(["cb_id", "subtheme_tag"])

childcare_categories_df


# ## Helper functions


# +
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


def save_data_table(table: pd.DataFrame, filename: str, folder):
    """Helper function to save table data underpinning figures"""
    table.to_csv(folder / f"{filename}.csv", index=False)


# -

# ## Analysis (all companies)

# Company identifiers
companies_ids = set(childcare_categories_df.cb_id.to_list())
companies_df = CB.cb_organisations.query("id in @companies_ids").pipe(
    select_by_role, "company"
)
# Funding data
funding_df = (
    CB.get_funding_rounds(companies_df)
    # .query("investment_type in @utils.EARLY_STAGE_DEALS")
)

# ### Composition

# +
# Total funding per company
companies_funding_df = funding_df.groupby("org_id", as_index=False).agg(
    total_funding_gbp=("raised_amount_gbp", "sum"),
)

companies_funding_theme_df = (
    companies_df.merge(companies_funding_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)
# -

columns_to_export = [
    "id",
    "name",
    "cb_url",
    "short_description",
    "country",
    "total_funding_gbp",
    "theme",
    "subtheme",
    "subtheme_full",
    "subtheme_tag",
]
companies_funding_theme_df[columns_to_export].to_csv(
    TABLES_DIR / "companies_funding_theme.csv", index=False
)

theme_counts_df = (
    childcare_categories_df.groupby(["theme"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    .sort_values(["count"], ascending=False)
)

theme_counts_df

# +
labels_label = "theme"
values_label = "count"
# tooltip = [labels_label, alt.Tooltip(values_label, format=",.3f")]

# %%
fig = (
    alt.Chart(
        theme_counts_df,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1500]),
            title="Number of companies",
        ),
        alt.Y(
            f"{labels_label}:N",
            sort="-x",
            title="",
        ),
        # tooltip=tooltip,
        color="theme",
    )
)
fig_final = pu.configure_plots(fig)
fig_final
# -

sort_order = theme_counts_df.theme.to_list()

subtheme_counts_df = (
    childcare_categories_df.groupby(["theme", "subtheme_full"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    # make theme categorical and provide sort_order list as the sorting order
    .astype({"theme": "category"})
    .assign(theme=lambda x: x.theme.cat.set_categories(sort_order))
    .sort_values(["theme", "count"], ascending=[True, False])
)
subtheme_counts_df

# +
labels_label = "subtheme_full"
values_label = "count"
# tooltip = [labels_label, alt.Tooltip(values_label, format=",.3f")]

# %%
fig = (
    alt.Chart(
        subtheme_counts_df,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1500]),
            title="Number of companies",
        ),
        alt.Y(
            f"{labels_label}:N",
            sort=subtheme_counts_df[labels_label].to_list(),
            title="",
            # increase label width
            axis=alt.Axis(labelLimit=1000),
        ),
        # tooltip=tooltip,
        color="theme",
    )
)
fig_final = pu.configure_plots(fig)
fig_final
# -

# ### Total investment over time, all companies

# Funding time series
funding_ts = au.cb_get_all_timeseries(
    companies_df, funding_df, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)

funding_ts

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
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
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q", scale=alt.Scale(domain=[0, 2500])),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final
# -

figure_name = f"parenting_tech_Total_investment"
AltairSaver.save(fig_final, figure_name, filetypes=filetypes)
save_data_table(data, figure_name, TABLES_DIR)

au.percentage_change(
    funding_ts.query("time_period == 2021").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

au.percentage_change(
    funding_ts.query("time_period == 2018").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

au.percentage_change(
    funding_ts.query("time_period == 2017").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

# ### Deal size time series

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
    .rename(
        columns={
            "time_period": horizontal_label,
            "no_of_rounds": values_label,
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
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q"),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final
# -

funding_df_ = funding_df.assign(
    deal_type=lambda df: df.raised_amount_gbp.apply(utils.deal_amount_to_range)
).assign(year=lambda df: pd.to_datetime(df.announced_on).dt.year)

deal_order = [
    "£0-1M",
    "£1-4M",
    "£4-15M",
    "£15-40M",
    "£40-100M",
    "£100-250M",
    "£250+" "n/a",
]

# altair stacked bar chart with number of deals, with deal types as color
data = (
    funding_df_.groupby(["year", "deal_type"], as_index=False)
    .agg(
        counts=("funding_round_id", "count"),
        total_amount=("raised_amount_gbp", "sum"),
    )
    .query("year > 2009")
    .query("year <= 2023")
    # convert to millions
    .assign(total_amount=lambda df: df.total_amount / 1000)
)


data

fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y("counts:Q", title="Number of deals"),
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

# normalized altair stacked bar chart with number of deals, with deal types as color
fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y(
            "counts:Q",
            title="Number of deals",
            stack="normalize",
            axis=alt.Axis(format="%"),
        ),
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y("total_amount:Q", title="Investment (£ millions)"),
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

# normalized altair stacked bar chart with number of deals, with deal types as color
fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y(
            "total_amount:Q",
            title="Investment (£ millions)",
            stack="normalize",
            axis=alt.Axis(format="%"),
        ),
        # sort legend
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

# ## By category

# +
themes_list = childcare_categories_df.theme.unique()
subthemes_list = childcare_categories_df.subtheme_full.unique()
themes_to_subthemes = utils.get_taxonomy_dict(
    childcare_categories_df, "theme", "subtheme_full"
)

themes_to_ids = {
    theme: set(childcare_categories_df.query("theme == @theme").cb_id)
    for theme in themes_list
}
subthemes_to_ids = {
    subtheme: set(childcare_categories_df.query("subtheme_full == @subtheme").cb_id)
    for subtheme in subthemes_list
}
# -

# Collect all theme time series (NB: Double counting will occur, so don't sum them)
themes_ts = []
for theme in themes_list:
    theme_ids = themes_to_ids[theme]
    # Get companies in a theme
    companies_theme_df = companies_df.query("id in @theme_ids").copy()
    # Get funding for companies in a theme
    funding_theme_df = (
        funding_df.query("org_id in @theme_ids").copy()
        # .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_theme_df, funding_theme_df, "year", 2010, 2023
    ).assign(theme=theme)
    # Collect all time series
    themes_ts.append(funding_ts)
# Final dataframe
themes_ts = pd.concat(themes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

# Collect all subtheme time series (NB: Double counting will occur, so don't sum them)
subthemes_ts = []
for subtheme in subthemes_list:
    subtheme_ids = subthemes_to_ids[subtheme]
    # Get companies in a theme
    companies_subtheme_df = companies_df.query("id in @subtheme_ids").copy()
    # Get funding for companies in a theme
    funding_subtheme_df = (
        funding_df.query("org_id in @subtheme_ids").copy()
        # .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_subtheme_df, funding_subtheme_df, "year", 2010, 2023
    ).assign(subtheme=subtheme)
    # Collect all time series
    subthemes_ts.append(funding_ts)
# Final dataframe
subthemes_ts = pd.concat(subthemes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

cats = themes_list
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        themes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="theme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

figure_name = f"parenting_tech_Total_investment_By_theme"
AltairSaver.save(fig, figure_name, filetypes=filetypes)
save_data_table(themes_ts, figure_name, TABLES_DIR)

# ## By category [early stage]

# Funding time series
funding_ts = au.cb_get_all_timeseries(
    companies_df,
    funding_df.query("investment_type in @utils.EARLY_STAGE_DEALS"),
    "year",
    2009,
    2023,
).assign(year=lambda df: df.time_period.dt.year)

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
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
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q", scale=alt.Scale(domain=[0, 2500])),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final
# -

figure_name = f"parenting_tech_Total_investment_Early_stage"
AltairSaver.save(fig_final, figure_name, filetypes=filetypes)
save_data_table(data, figure_name, TABLES_DIR)

au.percentage_change(
    funding_ts.query("time_period == 2021").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

au.percentage_change(
    funding_ts.query("time_period == 2020").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

# Collect all theme time series (NB: Double counting will occur, so don't sum them)
themes_ts = []
for theme in themes_list:
    theme_ids = themes_to_ids[theme]
    # Get companies in a theme
    companies_theme_df = companies_df.query("id in @theme_ids").copy()
    # Get funding for companies in a theme
    funding_theme_df = (
        funding_df.query("org_id in @theme_ids")
        .copy()
        .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_theme_df, funding_theme_df, "year", 2010, 2023
    ).assign(theme=theme)
    # Collect all time series
    themes_ts.append(funding_ts)
# Final dataframe
themes_ts = pd.concat(themes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

# Collect all subtheme time series (NB: Double counting will occur, so don't sum them)
subthemes_ts = []
for subtheme in subthemes_list:
    subtheme_ids = subthemes_to_ids[subtheme]
    # Get companies in a theme
    companies_subtheme_df = companies_df.query("id in @subtheme_ids").copy()
    # Get funding for companies in a theme
    funding_subtheme_df = (
        funding_df.query("org_id in @subtheme_ids")
        .copy()
        .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_subtheme_df, funding_subtheme_df, "year", 2010, 2023
    ).assign(subtheme=subtheme)
    # Collect all time series
    subthemes_ts.append(funding_ts)
# Final dataframe
subthemes_ts = pd.concat(subthemes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

cats = themes_list
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        themes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="theme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

figure_name = f"parenting_tech_Total_investment_By_theme_Early_stage"
AltairSaver.save(fig, figure_name, filetypes=filetypes)
save_data_table(themes_ts, figure_name, TABLES_DIR)

cats = themes_to_subthemes["Content"]
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subthemes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="subtheme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

cats = themes_to_subthemes["Family Support"]
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subthemes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="subtheme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

cats = themes_to_subthemes["Special Needs & Health"]
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subthemes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="subtheme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

# ## All deals, all countries

# altair stacked bar chart with number of deals, with deal types as color
data = (
    funding_df_.merge(companies_df[["id", "country"]], left_on="org_id", right_on="id")
    .query("year > 2018")
    .query("year <= 2022")
    .groupby(["country"], as_index=False)
    .agg(
        counts=("funding_round_id", "count"),
        total_amount=("raised_amount_gbp", "sum"),
    )
    # convert to millions
    .assign(total_amount=lambda df: df.total_amount / 1000)
    .sort_values("total_amount", ascending=False)
    .reset_index(drop=True)
)


data

# # UK

# +
funding_df_uk = (
    funding_df_.merge(companies_df[["id", "country"]], left_on="org_id", right_on="id")
    .query("country == 'United Kingdom'")
    .copy()
)
companies_df_uk = companies_df.query("country == 'United Kingdom'").copy()

# Funding time series
funding_ts_uk = au.cb_get_all_timeseries(
    companies_df_uk, funding_df_uk, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)
# -

funding_ts_uk

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts_uk.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
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
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q"),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final

# +
horizontal_label = "Year"
values_label = "Number of deals"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts_uk.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
    .rename(
        columns={
            "time_period": horizontal_label,
            "no_of_rounds": values_label,
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
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q"),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final

# +
# theme_counts_df = (
#     childcare_categories_df
#     .groupby(['theme'])
#     .size().reset_index().rename(columns={0: 'count'})
#     .sort_values(['count'], ascending=False)
# )
