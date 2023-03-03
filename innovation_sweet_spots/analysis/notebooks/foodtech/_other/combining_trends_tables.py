# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd
from innovation_sweet_spots.utils import chart_trends

TRENDS_DIR = PROJECT_DIR / "outputs/foodtech/trends"

# %%
df_guardian = pd.read_csv(TRENDS_DIR / "guardian_Report_Guardian_all.csv").assign(
    signal="News"
)
df_hansard = (
    pd.read_csv(TRENDS_DIR / "hansard_Report_Hansard_Categories.csv")
    .rename(columns={"tech_area": "Category"})
    .assign(signal="Parliament")
)
df_gtr_nihr = pd.read_csv(TRENDS_DIR / "research_Report_GTR_NIHR_all.csv").assign(
    signal="Research"
)
df_vc = (
    pd.read_csv(TRENDS_DIR / "venture_capital_Report_VC_all.csv")
    .drop(["Growth", "Number of deals", "Number of companies"], axis=1)
    .rename(columns={"Magnitude": "magnitude"})
    .assign(signal="Investment")
)
df_vc.loc[df_vc["Sub Category"] == "Alt protein (all)", "Sub Category"] = "Alt protein"


# %%
trends_combined = (
    pd.concat(
        [
            df_guardian,
            df_hansard,
            df_gtr_nihr,
            df_vc,
        ],
        ignore_index=False,
    )
    .drop(["index", "trend_type_suggestion"], axis=1)
    .fillna("n/a (category level)")
)
trends_combined.loc[
    trends_combined["Sub Category"] == "n/a (category level)", "Sub Category"
] = "na"


# %%
trends_combined.to_csv(
    PROJECT_DIR / "outputs/foodtech/trends/trends_combined.csv", index=False
)

# %%
subcats = ["Reformulation", "Alt protein"]
cats = [
    "Logistics",
    "Food waste",
    "Cooking and kitchen",
    "Health",
    "Restaurants and retail",
]

trends_map_raw = trends_combined.query(
    '`Sub Category`in @subcats or ((`Category` in @cats) and (`Sub Category` == "na"))'
)

# %%
subcats = [
    "Reformulation",
    "Alt protein",
    "Delivery",
    "Meal kits",
    "Personalised nutrition",
    "Biomedical",
    "Kitchen tech",
]
cats = ["Food waste", "Restaurants and retail"]

trends_map_raw = trends_combined.query(
    '`Sub Category`in @subcats or ((`Category` in @cats) and (`Sub Category` == "na"))'
)

# %%
trends_map = []
for signal in trends_map_raw.signal.unique():
    trends_map.append(
        chart_trends.estimate_trend_type(
            trends_map_raw.query("signal == @signal"),
            magnitude_column="magnitude",
            growth_column="growth",
        )
    )
trends_map = pd.concat(trends_map, ignore_index=True).sort_values(
    ["signal", "Category", "Sub Category"]
)

# %%
trends_map

# %%
signal = "Research"
df_plot = trends_map.query("signal == @signal").assign(
    text=lambda df: df["Category"]
    + "/"
    + df["Sub Category"]
    + ":"
    + df["trend_type_suggestion"]
)

# %%
chart_trends._epsilon = 0.02
fig = chart_trends.mangitude_vs_growth_chart(
    data=df_plot,
    x_limit=24,
    y_limit=5,
    mid_point=df_plot.magnitude.median(),
    baseline_growth=0,
    values_label="Magnitude",
    text_column="text",
)
fig.interactive()

# %%
signal = "Investment"
df_plot = trends_map.query("signal == @signal").assign(
    text=lambda df: df["Category"]
    + "/"
    + df["Sub Category"]
    + ":"
    + df["trend_type_suggestion"]
)

# %%
df_plot

# %%
chart_trends._epsilon = 0.01
fig = chart_trends.mangitude_vs_growth_chart(
    data=df_plot,
    x_limit=12_000,
    y_limit=12,
    mid_point=df_plot.magnitude.median(),
    baseline_growth=0,
    values_label="Magnitude",
    text_column="text",
)
fig.interactive()

# %%
