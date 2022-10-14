"""
"""
import innovation_sweet_spots.analysis.analysis_utils as au
import pandas as pd
from collections import defaultdict
import itertools
import numpy as np

# EXCLUDED_DEAL_TYPES = [
#     "GRANT",
#     '-',
#     # np.nan,
#     'ICO',
# ]

EARLY_DEAL_TYPES = [
    "SERIES B",
    "EARLY VC",
    "SERIES A",
    "SEED",
    "ANGEL",
    "CONVERTIBLE",
    "LATE VC",
    "SPINOUT",
    "GROWTH EQUITY VC",
    "SERIES C",
    "SERIES D",
    "PRIVATE PLACEMENT VC",
    "SERIES F",
    "SERIES E",
    "SECONDARY",
    "SERIES H",
    "SERIES G",
    "SERIES I",
    "SPAC PRIVATE PLACEMENT",
    "GROWTH EQUITY NON VC",
    "MEDIA FOR EQUITY",
    "PROJECT, REAL ESTATE, INFRASTRUCTURE FINANCE",
]

LATE_DEAL_TYPES = [
    "ACQUISITION",
    "DEBT",
    "POST IPO EQUITY",
    "IPO",
    "MERGER",
    "SPAC IPO",
    "BUYOUT",
    "POST IPO DEBT",
    "POST IPO CONVERTIBLE",
    "LENDING CAPITAL",
    "POST IPO SECONDARY",
]

rejected_tags = ["pet food", "pet care", "pet", "veterinary"]


def result_dict_to_dataframe(
    result_dict: dict, sort_by: str = "counts", category_name: str = "cluster"
) -> pd.DataFrame:
    """Prepares the output dataframe"""
    return (
        pd.DataFrame(result_dict)
        .T.reset_index()
        .sort_values(sort_by)
        .rename(columns={"index": category_name})
    )


def get_category_time_series(
    time_series_df: pd.DataFrame,
    category_of_interest: str,
    time_column: str = "releaseYear",
    category_column: str = "cluster",
) -> dict:
    """Gets cluster or user-specific time series"""
    return (
        time_series_df.query(f"{category_column} == @category_of_interest")
        .drop(category_column, axis=1)
        .sort_values(time_column)
        .rename(columns={time_column: "year"})
        .assign(year=lambda x: pd.to_datetime(x.year.apply(lambda y: str(int(y)))))
        .pipe(
            au.impute_empty_periods,
            time_period_col="year",
            period="Y",
            min_year=2010,
            max_year=2021,
        )
        .assign(year=lambda x: x.year.dt.year)
    )


def get_estimates(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
    estimate_function=au.growth,
    year_start: int = 2019,
    year_end: int = 2020,
):
    """
    Get growth estimate for each category

    growth_estimate_function - either growth, smoothed_growth, or magnitude
    For growth, use 2019 and 2020 as year_start and year_end
    For smoothed_growth and magnitude, use 2017 and 2021
    """
    time_series_df_ = time_series_df[[time_column, category_column, value_column]]

    result_dict = {
        category: estimate_function(
            get_category_time_series(
                time_series_df_, category, time_column, category_column
            ),
            year_start=year_start,
            year_end=year_end,
        )
        for category in time_series_df[category_column].unique()
    }
    return result_dict_to_dataframe(result_dict, value_column, category_column)


def get_magnitude_vs_growth(
    time_series_df: pd.DataFrame,
    value_column: str = "counts",
    time_column: str = "releaseYear",
    category_column: str = "cluster",
    year_start=2017,
    year_end=2021,
):
    """Get magnitude vs growth esitmates"""
    df_growth = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=au.smoothed_growth,
        year_start=year_start,
        year_end=year_end,
    ).rename(columns={value_column: "Growth"})

    df_magnitude = get_estimates(
        time_series_df,
        value_column=value_column,
        time_column=time_column,
        category_column=category_column,
        estimate_function=au.magnitude,
        year_start=year_start,
        year_end=year_end,
    ).rename(columns={value_column: "Magnitude"})

    return df_growth.merge(df_magnitude, on=category_column)


def deal_amount_to_range(
    amount: float, currency: str = "£", categories: bool = True
) -> str:
    """
    Convert amounts to range in millions
    Args:
        amount: Investment amount (in GBP thousands)
        categories: If True, adding indicative deal categories
        currency: Currency symbol
    """
    amount /= 1e3
    if (amount >= 0.001) and (amount < 1):
        return f"{currency}0-1M" if not categories else f"{currency}0-1M"
    elif (amount >= 1) and (amount < 4):
        return f"{currency}1-4M" if not categories else f"{currency}1-4M"
    elif (amount >= 4) and (amount < 15):
        return f"{currency}4-15M" if not categories else f"{currency}4-15M"
    elif (amount >= 15) and (amount < 40):
        return f"{currency}15-40M" if not categories else f"{currency}15-40M"
    elif (amount >= 40) and (amount < 100):
        return f"{currency}40-100M" if not categories else f"{currency}40-100M"
    elif (amount >= 100) and (amount < 250):
        return f"{currency}100-250M"
    elif amount >= 250:
        return f"{currency}250+"
    else:
        return "n/a"


def get_category_ids_(taxonomy_df, rejected_tags, DR, column="Category"):
    category_ids = defaultdict(set)

    rejected_ids = [
        DR.get_ids_by_labels(row.Category, row.label_type)
        for i, row in DR.labels.query("Category in @rejected_tags").iterrows()
    ]
    rejected_ids = set(itertools.chain(*rejected_ids))

    for category in taxonomy_df[column].unique():
        ids = [
            DR.get_ids_by_labels(row.Category, row.label_type)
            for i, row in taxonomy_df.query(f"`{column}` == @category").iterrows()
        ]
        ids = set(itertools.chain(*ids)).difference(rejected_ids)
        category_ids[category] = ids
    return category_ids


def get_category_ids(
    taxonomy_df, rejected_tags, company_to_taxonomy_df, DR, column="Minor"
):
    category_ids = defaultdict(set)

    rejected_ids = [
        DR.get_ids_by_labels(row.Category, row.label_type)
        for i, row in DR.labels.query("Category in @rejected_tags").iterrows()
    ]
    rejected_ids = set(itertools.chain(*rejected_ids))

    for category in taxonomy_df[column].unique():
        ids = (
            company_to_taxonomy_df.query("level == @column")
            .query("Category == @category")
            .id.to_list()
        )
        ids = set(ids).difference(rejected_ids)
        category_ids[category] = ids
    return category_ids


def get_category_ts(category_ids, DR, deal_type=EARLY_DEAL_TYPES):
    ind_ts = []
    for category in category_ids:
        ids = category_ids[category]
        ind_ts.append(
            au.cb_get_all_timeseries(
                DR.company_data.query("id in @ids"),
                (
                    DR.funding_rounds.query("id in @ids").query(
                        "`EACH ROUND TYPE` in @deal_type"
                    )
                ),
                period="year",
                min_year=2010,
                max_year=2022,
            )
            .assign(year=lambda df: df.time_period.dt.year)
            .assign(Category=category)
        )
    return pd.concat(ind_ts, ignore_index=True)


def get_company_counts(category_ids: dict):
    return pd.DataFrame(
        [(key, len(np.unique(list(category_ids[key])))) for key in category_ids],
        columns=["Category", "Number of companies"],
    )


def get_deal_counts(DR, category_ids: dict):
    category_deal_counts = []
    for key in category_ids:
        ids = category_ids[key]
        deals = DR.funding_rounds.query("id in @ids").query(
            "`EACH ROUND TYPE` in @EARLY_DEAL_TYPES"
        )
        category_deal_counts.append((key, len(deals)))
    return pd.DataFrame(category_deal_counts, columns=["Category", "Number of deals"])


def get_trends(
    taxonomy_df,
    rejected_tags,
    taxonomy_level,
    company_to_taxonomy_df,
    DR,
    deal_type=EARLY_DEAL_TYPES,
):
    category_ids = get_category_ids(
        taxonomy_df, rejected_tags, company_to_taxonomy_df, DR, taxonomy_level
    )
    company_counts = get_company_counts(category_ids)
    category_ts = get_category_ts(category_ids, DR, deal_type)

    values_title_ = "raised_amount_gbp_total"
    values_title = "Growth"
    category_title = "Category"
    colour_title = category_title
    horizontal_title = "year"

    if taxonomy_level == "Category":
        tax_levels = ["Category", "Minor", "Major"]
    if taxonomy_level == "Minor":
        tax_levels = ["Minor", "Major"]
    if taxonomy_level == "Major":
        tax_levels = ["Major"]

    return (
        get_magnitude_vs_growth(
            category_ts,
            value_column=values_title_,
            time_column=horizontal_title,
            category_column=category_title,
        )
        .assign(growth=lambda df: df.Growth / 100)
        .merge(get_deal_counts(DR, category_ids), on="Category")
        .merge(company_counts, on="Category")
        .merge(
            taxonomy_df[tax_levels].drop_duplicates(taxonomy_level),
            how="left",
            left_on="Category",
            right_on=taxonomy_level,
        )
    )


## Figure functions
import altair as alt


def fig_growth_vs_magnitude(
    magnitude_vs_growth,
    colour_field,
    text_field,
    legend=alt.Legend(),
    horizontal_scale="log",
):
    title_text = "Foodtech trends (2017-2021)"
    subtitle_text = [
        "Data: Dealroom. Showing data on early stage deals (eg, series funding)",
        "Late stage deals, such as IPOs, acquisitions, and debt not included.",
    ]

    fig = (
        alt.Chart(
            magnitude_vs_growth,
            width=400,
            height=400,
        )
        .mark_circle(size=50)
        .encode(
            x=alt.X(
                "Magnitude:Q",
                axis=alt.Axis(title=f"Average yearly raised amount (million GBP)"),
                # scale=alt.Scale(type="linear"),
                scale=alt.Scale(type=horizontal_scale),
            ),
            y=alt.Y(
                "growth:Q",
                axis=alt.Axis(title="Growth", format="%"),
                # axis=alt.Axis(
                #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
                # ),
                # scale=alt.Scale(domain=(-.100, .300)),
                #             scale=alt.Scale(type="log", domain=(.01, 12)),
            ),
            #         size="Number of companies:Q",
            color=alt.Color(f"{colour_field}:N", legend=legend),
            tooltip=[
                "Category",
                alt.Tooltip(
                    "Magnitude", title=f"Average yearly raised amount (million GBP)"
                ),
                alt.Tooltip("growth", title="Growth", format=".0%"),
            ],
        )
        .properties(
            title={
                "anchor": "start",
                "text": title_text,
                "subtitle": subtitle_text,
                "subtitleFont": pu.FONT,
            },
        )
    )

    text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
        text=text_field
    )

    fig_final = (
        (fig + text)
        .configure_axis(
            grid=False,
            gridDash=[1, 7],
            gridColor="white",
            labelFontSize=pu.FONTSIZE_NORMAL,
            titleFontSize=pu.FONTSIZE_NORMAL,
        )
        .configure_legend(
            titleFontSize=pu.FONTSIZE_NORMAL,
            labelFontSize=pu.FONTSIZE_NORMAL,
        )
        .configure_view(strokeWidth=0)
    )

    return fig_final


def fig_category_growth(
    magnitude_vs_growth_filtered,
    colour_field,
    text_field,
    height=500,
):
    """ """
    fig = (
        alt.Chart(
            (
                magnitude_vs_growth_filtered.assign(
                    Increase=lambda df: df.growth > 0
                ).assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
            ),
            width=300,
            height=height,
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
                #             scale=alt.Scale(domain=(-1, 37)),
            ),
            y=alt.Y(
                "Category:N",
                sort="-x",
                axis=alt.Axis(title="Category", labels=False),
            ),
            size=alt.Size(
                "Magnitude",
                title="Yearly investment (million GBP)",
                legend=alt.Legend(orient="top"),
                scale=alt.Scale(domain=[100, 4000]),
            ),
            color=alt.Color(
                colour_field,
            ),
            # size="cluster_size:Q",
            #         color=alt.Color(f"{colour_title}:N", legend=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip(
                    "Magnitude:Q",
                    format=",.3f",
                    title="Average yearly investment (million GBP)",
                ),
                "Number of companies",
                "Number of deals",
                alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
            ],
        )
    )

    text = (
        alt.Chart(magnitude_vs_growth_filtered)
        .mark_text(align="left", baseline="middle", font=pu.FONT, dx=7)
        .encode(
            text=text_field,
            x="growth:Q",
            y=alt.Y("Category:N", sort="-x"),
        )
    )

    # text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
    #     text='text_label:N'
    # )

    # fig_final = (
    #     (fig + text)
    #     .configure_axis(
    #         gridDash=[1, 7],
    #         gridColor="grey",
    #         labelFontSize=pu.FONTSIZE_NORMAL,
    #         titleFontSize=pu.FONTSIZE_NORMAL,
    #     )
    #     .configure_legend(
    #         labelFontSize=pu.FONTSIZE_NORMAL - 1,
    #         titleFontSize=pu.FONTSIZE_NORMAL - 1,
    #     )
    #     .configure_view(strokeWidth=0)
    #     #     .interactive()
    # )

    return pu.configure_titles(pu.configure_axes((fig + text)), "", "")


def fig_size_vs_magnitude(
    magnitude_vs_growth_filtered,
    colour_field,
    horizontal_scale="log",
):
    fig = (
        alt.Chart(
            magnitude_vs_growth_filtered,
            width=500,
            height=450,
        )
        .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1, size=50)
        .encode(
            x=alt.X(
                "Number of companies:Q",
            ),
            y=alt.Y(
                "Magnitude:Q",
                axis=alt.Axis(title=f"Average yearly raised amount (million GBP)"),
                scale=alt.Scale(type=horizontal_scale),
            ),
            color=alt.Color(colour_field),
            # size="cluster_size:Q",
            #         color=alt.Color(f"{colour_title}:N", legend=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip(
                    "Magnitude:Q",
                    format=",.3f",
                    title="Average yearly investment (million GBP)",
                ),
                alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
                "Number of companies",
                "Number of deals",
            ],
        )
    )

    return pu.configure_titles(pu.configure_axes(fig), "", "")
