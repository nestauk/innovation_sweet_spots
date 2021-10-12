import altair as alt
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.topic_analysis as iss_topics
import pandas as pd
import numpy as np

### COLOR SCHEMES
PLOT_LCH_CATEGORY_DOMAIN = [
    "Heat pumps",
    "Biomass heating",
    "Hydrogen heating",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Heat storage",
    "Insulation & retrofit",
    "Energy management",
]

PLOT_LCH_CATEGORY_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    #     '#bab0ac',
]
COLOR_PAL_LCH = (PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS)

PLOT_REF_CATEGORY_DOMAIN = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]

PLOT_REF_CATEGORY_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    #     '#bab0ac',
]

COLOR_PAL_REF = (PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS)

### CALCULATIONS
def get_growth_and_level(
    YEARLY_STATS, cat, variable, year_1=2016, year_2=2020, window=3
):
    df = YEARLY_STATS[cat].copy()
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    if df_ma.loc[year_1, variable] != 0:
        growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    else:
        growth_rate = np.nan
    level = df.loc[year_1:year_2, variable].mean()
    return growth_rate, level


def get_growth_and_level_std(
    YEARLY_STATS, cat, variable, year_1=2016, year_2=2020, window=3
):
    df = YEARLY_STATS[cat].copy()
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    if df_ma.loc[year_1, variable] != 0:
        growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    else:
        growth_rate = np.nan
    level = df.loc[year_1:year_2, variable].mean()
    level_std = df.loc[year_1:year_2, variable].std()
    return growth_rate, level, level_std


def get_year_by_year_stats(YEARLY_STATS, variable, year_dif=4):
    df_stats_all = pd.DataFrame()
    for cat in list(YEARLY_STATS.keys()):
        for y in range(2011, 2021):
            yy = year_dif
            #         if y==2009: yy=1;
            #         if y==2010: yy=2;
            #         if y==2011: yy=2;
            #         if y>2010: yy=4;
            df_stats = pd.DataFrame(
                [
                    get_growth_and_level(cat, variable, y - yy, y) + tuple([cat, y])
                    for cat in [cat]
                ],
                columns=["growth", variable, "tech_category", "year"],
            )
            df_stats_all = df_stats_all.append(df_stats, ignore_index=True)
    #     df_stats_all.loc[df_stats_all.tech_category==cat, variable] = df_stats_all.loc[df_stats_all.tech_category==cat, variable] / df_stats_all[(df_stats_all.tech_category==cat)].iloc[0][variable]
    return df_stats_all


### FIGURES
def plot_bars(
    YEARLY_STATS, variable, cat, y_label, final_year=2020, bar_color="#2F1847"
):
    """ """
    df_ = YEARLY_STATS[cat]
    df_ = df_[df_.year <= final_year]
    ww = 350
    hh = 210
    base = alt.Chart(df_, width=ww, height=hh).encode(
        alt.X(
            "year:O",
            #             axis=alt.Axis(title=None, labels=False)
        )
    )
    fig_projects = base.mark_bar(color=bar_color, size=20).encode(
        alt.Y(variable, axis=alt.Axis(title=y_label, titleColor=bar_color))
    )
    return iss.nicer_axis(fig_projects)


def plot_matrix(
    YEARLY_STATS,
    variable,
    category_names,
    x_label,
    y_label="Growth",
    year_1=2016,
    year_2=2020,
    color_pal=(PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS),
    window=3,
    max_x=None,
    max_y=None,
):

    df_stats = pd.DataFrame(
        [
            get_growth_and_level(
                YEARLY_STATS, cat, variable, year_1, year_2, window=window
            )
            + tuple([cat])
            for cat in category_names
        ],
        columns=["growth", variable, "tech_category"],
    )

    if max_y is None:
        max_y = df_stats.growth.max()
    if max_x is None:
        max_x = df_stats[variable].max()

    points = (
        alt.Chart(
            df_stats,
            height=350,
            width=350,
        )
        .mark_circle(size=35)
        .encode(
            alt.Y("growth:Q", title=y_label, scale=alt.Scale(domain=(0, max_y))),
            alt.X(f"{variable}:Q", title=x_label, scale=alt.Scale(domain=(0, max_x))),
            color=alt.Color(
                "tech_category",
                scale=alt.Scale(domain=color_pal[0], range=color_pal[1]),
                #             color=alt.Color(
                #                 'tech_category',
                #                 scale=alt.Scale(domain=category_names,
                #                                 range=COLOUR_PAL[0:len(category_names)]),
                legend=None,
            ),
        )
    )

    text = points.mark_text(align="left", baseline="middle", dx=7, size=15).encode(
        text="tech_category"
    )

    fig = (
        (points + text)
        .configure_axis(grid=True)
        .configure_view(strokeOpacity=1, strokeWidth=2)
    )
    return fig
