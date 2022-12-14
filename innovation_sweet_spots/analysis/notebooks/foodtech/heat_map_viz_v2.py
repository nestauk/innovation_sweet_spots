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
# # Sketching the heat map visualisation

# %%
import altair as alt
import pandas as pd
from innovation_sweet_spots.utils import plotting_utils as pu
from innovation_sweet_spots.getters import google_sheets

# %%
import importlib

importlib.reload(google_sheets)

# %%
# Functionality for saving charts
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %%
df_melt = (
    google_sheets.get_foodtech_heat_map(
        from_local=False, data_range="heatmap_data_final"
    )
    .query("Include == '1'")
    .astype({"Trend_index": float, "Trend_index_final": float, "weight": float})
)

# %%
trend_index = (
    df_melt
    # .assign(Trend_index_final=lambda df: df.Trend_index_final * df.weight)
    .groupby("Category", as_index=False)
    .agg(Trend_index_sum=("Trend_index_final", "sum"))
    .sort_values(["Trend_index_sum", "Category"], ascending=False)
    .assign(Trend_index_final=lambda df: df.Trend_index_sum / 4.5)
)

# %%
trend_index.round(2)

# %%
value_to_trend = {
    "1": "Dormant",
    "2": "Stabilising",
    "3": "Emerging",
    "4": "Hot",
}

trend_to_value = {
    "Dormant": 1,
    "Stabilising": 2,
    "Emerging": 3,
    "Hot": 4,
}


# %%
# Colour scale
domain_ = ["Dormant", "Stabilising", "Emerging", "Hot"]
range_ = [
    pu.NESTA_COLOURS[11],
    pu.NESTA_COLOURS[8],
    pu.NESTA_COLOURS[1],
    pu.NESTA_COLOURS[4],
]

# size_values = "Size"
size_values = "Magnitude_scaled"

signal_sort_order = ["Investment", "Research", "News", "Parliament"]
category_sort_order = trend_index.Category.to_list()
# category_sort_order = [
#     "Innovative food: Alternative protein",
#     "Logistics",
#     "Food waste",
#     "Innovative food: Reformulation",
#     "Restaurants and retail",
#     "Personalised nutrition",
#     "Cooking and kitchen tech",
# ]

# %%
# Heat map
selection = alt.selection_single(empty="none")

fig = (
    alt.Chart(df_melt, width=300, height=300)
    .mark_square(size=300)
    .encode(
        x=alt.X(
            "Signal:N",
            title=None,
            axis=alt.Axis(labelAngle=0, domain=False, ticks=False, orient="top"),
            sort=signal_sort_order,
        ),
        y=alt.Y(
            "Category:N",
            title=None,
            sort=category_sort_order,
            axis=alt.Axis(labelLimit=300, domain=False, ticks=False),
        ),
        color=alt.Color(
            "Trend:O",
            legend=alt.Legend(orient="right"),
            scale=alt.Scale(domain=domain_, range=range_),
        ),
        # size=alt.Size(size_values, legend=None, scale=alt.Scale(range=[25, 700])),
        tooltip=["Trend", "Comment"],
    )
).add_selection(selection)

data_text = df_melt.assign(
    text_x=1,
    text_y=5,
)

text = (
    alt.Chart(data_text)
    .mark_text()
    .encode(
        x=alt.X("text_x:Q"),
        y=alt.Y("text_y:Q"),
        text="Comment",
    )
    .transform_filter(selection)
)

# (fig + text)

fig = pu.configure_plots(fig)
fig

# %%
from innovation_sweet_spots import PROJECT_DIR

df_melt.to_csv(PROJECT_DIR / "outputs/foodtech/interim/heatmap_data.csv", index=False)

# %%
AltairSaver.save(fig, f"v2022_11_18_Heat_map", filetypes=["html", "svg", "png"])

# %% [markdown]
# ## Experimenting with text comments

# %%
import pandas as pd
from urllib.parse import urlencode


def make_google_query(name):
    return "https://www.google.com/search?" + urlencode({"q": '"{0}"'.format(name)})


data = [
    [
        "GO:0005874",
        "microtubule",
        0.590923058896654,
        -4.00372136407618,
        3.93380303408685,
        5.07433362293908,
        0.743307611944267,
        0,
    ],
    [
        "GO:0042555",
        "MCM complex",
        0.0516392267501353,
        5.29409032883786,
        -0.438484234906433,
        4.01582063426207,
        0.736867388621876,
        0,
    ],
    [
        "GO:0005886",
        "plasma membrane",
        15.5064680247866,
        -2.59930578712986,
        -4.79438349762051,
        6.49331205332051,
        0.980465972776413,
        4.124e-05,
    ],
    [
        "GO:0030173",
        "integral component of Golgi membrane",
        0.0482779463204013,
        -0.0820996416106789,
        6.61844221537962,
        3.98659260682221,
        0.720783016873817,
        0.16417986,
    ],
    [
        "GO:0031083",
        "BLOC-1 complex",
        0.0157955281823943,
        6.03044083325888,
        2.61728943021364,
        3.50147007210041,
        0.638408624494431,
        0.22740185,
    ],
    [
        "GO:0030532",
        "small nuclear ribonucleoprotein complex",
        0.138166054523554,
        2.1939043417736,
        2.03060434260059,
        4.44321603416583,
        0.571526896999077,
        0.2622474,
    ],
    [
        "GO:0008250",
        "oligosaccharyltransferase complex",
        0.0394539627330108,
        2.22238070210506,
        4.52148800747906,
        3.89894446686651,
        0.602647357590838,
        0.39260902,
    ],
]

columns = [
    "term_ID",
    "description",
    "frequency",
    "plot_X",
    "plot_Y",
    "log_size",
    "uniqueness",
    "dispensability",
]

df = pd.DataFrame(data, columns=columns)
df["url"] = df["term_ID"].apply(make_google_query)

selection = alt.selection_single(empty="none")

base = (
    alt.Chart(df)
    .mark_point(filled=True, fillOpacity=0.5)
    .encode(
        y=alt.Y("plot_Y", title="Semantic Space Y"),
        x=alt.X("plot_X", title="Semantic Space X"),
        size=alt.Size(
            "log_size", scale=alt.Scale(base=0, domain=[3.5, 6], range=[1000, 7000])
        ),
        color=alt.Color("uniqueness", scale=alt.Scale(scheme="viridis")),
        tooltip=[
            alt.Tooltip("description"),
            alt.Tooltip("frequency"),
            alt.Tooltip("log_size"),
            alt.Tooltip("uniqueness"),
            alt.Tooltip("dispensability"),
        ],
    )
    .add_selection(selection)
)


df["long_description"] = (
    "description: "
    + df["description"]
    + "\nuniqueness: "
    + df["uniqueness"].astype(str)
).str.split(
    "\n"
)  # Create a list to split into multiple lines in the chart

text = (
    alt.Chart(df)
    .mark_text()
    .encode(
        y=alt.Y("plot_Y", title="Semantic Space Y"),
        x=alt.X("plot_X", title="Semantic Space X"),
        color=alt.Color("uniqueness", scale=alt.Scale(scheme="viridis")),
        text="long_description",
    )
    .transform_filter(selection)
)

(base + text).properties(title="plot", height=600, width=800).interactive()

# %%
