"""
innovation_sweet_spots.utils.plotting_utils
Functions for generating graphs
"""
import innovation_sweet_spots.analysis.analysis_utils as au
import altair as alt
import pandas as pd

ChartType = alt.vegalite.v4.api.Chart

# Brand aligned fonts and colours
FONT = "Averta"
TITLE_FONT = "Zosia"
NESTA_COLOURS = [
    "#0000FF",
    "#FDB633",
    "#18A48C",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#FFFFFF",
    "#000000",
]

# Investment deal categories
CURRENCY = "£"
DEAL_CATEGORIES = [
    f"{CURRENCY}0-1M",
    f"{CURRENCY}1-4M",
    f"{CURRENCY}4-15M",
    f"{CURRENCY}15-40M",
    f"{CURRENCY}40-100M",
    f"{CURRENCY}100-250M",
    f"{CURRENCY}250+",
    "n/a",
]
DEAL_CATEGORIES_ = [
    f"{CURRENCY}0-1M (pre-seed)",
    f"{CURRENCY}1-4M (seed)",
    f"{CURRENCY}4-15M (series A)",
    f"{CURRENCY}15-40M (series B)",
    f"{CURRENCY}40-100M (series C)",
    f"{CURRENCY}100-250M",
    f"{CURRENCY}250+",
    "n/a",
]


def nestafont():
    """Define Nesta fonts"""
    return {
        "config": {
            "title": {"font": TITLE_FONT, "anchor": "start"},
            "subtitle": {"font": FONT},
            "axis": {"labelFont": FONT, "titleFont": FONT},
            "header": {"labelFont": FONT, "titleFont": FONT},
            "legend": {"labelFont": FONT, "titleFont": FONT},
            "range": {
                "category": NESTA_COLOURS,
                "ordinal": {
                    "scheme": NESTA_COLOURS
                },  # this will interpolate the colors
            },
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")


def test_chart():
    """Generates a simple test chart"""
    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "labels": ["A", "B", "C"],
                    "values": [10, 15, 30],
                    "label": ["This is A", "This is B", "And this is C"],
                }
            ),
            width=400,
            height=200,
        )
        .mark_bar()
        .encode(
            alt.Y("labels:O", title="Vertical axis"),
            alt.X("values:Q", title="Horizontal axis"),
            tooltip=["label", "values"],
            color="labels",
        )
        .properties(
            title={
                "anchor": "start",
                "text": ["Chart title"],
                "subtitle": ["Longer descriptive subtitle"],
                "subtitleFont": FONT,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )


def process_axis_label(text: str, units: str = None):
    """Turns snake case into normal text and capitalises first letter"""
    new_text = " ".join(text.split("_")).capitalize()
    if units is not None:
        new_text += f" ({units})"
    return new_text


def convert_time_period(dates: pd.Series, period: str) -> list:
    """Converts datetimes to years, quarters or months for plotting"""
    if period == "Y":
        return list(dates.dt.year)
    elif period == "Q":
        return [f"{row.year}-Q{row.quarter}" for row in dates]
    elif period == "M":
        return [f'{row.year}-{row.strftime("%m")}' for row in dates]


def time_series(
    data: pd.DataFrame,
    y_column: str,
    y_units: str = None,
    x_column: str = "time_period",
    period: str = "Y",
    show_trend: bool = False,
) -> ChartType:
    """Basic time series plot"""
    chart = (
        alt.Chart(
            data.assign(**{x_column: convert_time_period(data[x_column], period)}),
            width=400,
            height=200,
        )
        .mark_line(point=False, stroke=NESTA_COLOURS[0], strokeWidth=1.5)
        .encode(alt.X(f"{x_column}:O"), alt.Y(f"{y_column}:Q"))
    )
    chart.encoding.x.title = process_axis_label(x_column)
    chart.encoding.y.title = process_axis_label(y_column, units=y_units)
    if show_trend:
        chart_trend = chart.transform_regression(x, y).mark_line(
            stroke="black", strokeDash=[2, 2], strokeWidth=0.5
        )
        chart = alt.layer(chart, chart_trend)
    return standardise_chart(chart)


def standardise_chart(chart: ChartType) -> ChartType:
    return chart.configure_axis(
        gridDash=[1, 6],
        gridColor="grey",
    ).configure_view(strokeWidth=0)


def cb_investments_barplot(
    data: pd.DataFrame,
    y_column: str,
    y_label: str = None,
    y_units: str = None,
    y_max: float = None,
    x_column: str = "time_period",
    x_label: str = None,
    period: str = "Y",
    show_trend: bool = False,
):
    """Barplot"""
    y_max = data[y_column].max() if y_max is None else y_max
    chart = (
        alt.Chart(
            data.assign(**{x_column: convert_time_period(data[x_column], period)}),
            width=400,
            height=200,
        )
        .mark_bar(color=NESTA_COLOURS[0])
        .encode(
            x=alt.X(f"{x_column}:O"),
            y=alt.Y(f"{y_column}:Q", scale=alt.Scale(domain=[0, y_max])),
            tooltip=[x_column, y_column],
        )
    )
    chart.encoding.x.title = (
        process_axis_label(x_column) if x_label is None else x_label
    )
    chart.encoding.y.title = (
        process_axis_label(y_column, units=y_units) if y_label is None else y_label
    )
    return standardise_chart(chart)


def cb_deal_types(
    funding_rounds: pd.DataFrame,
    simpler_types: bool = False,
    deal_types: list = None,
    stack: str = None,
) -> ChartType:
    """
    Stacked bar chart of the investment deal types per year
    Args:
        funding_rounds: Dataframe with funding round data
        simpler_types: Use a simpler, Dealroom-style categorisation
        deal_types: List of permitted deal types
        stack: Either None or 'normalize'
    """
    df = funding_rounds.copy()
    # Simplify deal categories
    if simpler_types:
        df["investment_type"] = df["raised_amount_gbp"].apply(
            au.cb_deal_amount_to_range
        )
    # Select only permitted deal types
    df = df if deal_types is None else df.query("investment_type in @deal_types")
    # Plot the chart
    return standardise_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(
                "count(investment_type)",
                title="Number of deals",
                stack=stack,
            ),
            color=alt.Color("investment_type", sort=DEAL_CATEGORIES_),
        )
        .configure_legend(title=None)
    )


def infer_geo_label(category_column: str) -> str:
    return category_column.split("_")[-1].capitalize()


def cb_top_geographies(
    data: pd.DataFrame,
    value_column: str,
    category_column: str = "org_country",
    value_label: str = None,
    category_label: str = None,
    top_n: int = 10,
) -> ChartType:
    """
    Plots a bar chart with stats for top_n countries
    Args:
        data: Dataframe with a columns for 'country'
    """
    category_label = (
        infer_geo_label(category_column) if category_label is None else category_label
    )
    df = data.sort_values(value_column, ascending=False).head(top_n)
    chart = (
        alt.Chart(df)
        .mark_bar(color=NESTA_COLOURS[0])
        .encode(
            y=alt.Y(category_column, sort="-x", title=category_label),
            x=alt.X(value_column),
        )
    )
    chart.encoding.x.title = (
        process_axis_label(value_column) if value_label is None else value_label
    )
    chart.encoding.y.title = (
        process_axis_label(category_column)
        if category_label is None
        else category_label
    )
    return standardise_chart(chart)


def time_series_by_category(
    data: pd.DataFrame,
    value_column: str,
    value_units: str = None,
    time_column: str = "time_period",
    period: str = "Y",
    category_column: str = "geography",
) -> ChartType:
    """Basic time series plot"""
    chart = (
        alt.Chart(
            data.assign(
                **{time_column: convert_time_period(data[time_column], period)}
            ),
            width=400,
            height=200,
        )
        .mark_line(point=False, strokeWidth=1.5)
        .encode(
            alt.X(f"{time_column}:O"),
            alt.Y(f"{value_column}:Q"),
            color=category_column,
        )
    )
    chart.encoding.x.title = process_axis_label(time_column)
    chart.encoding.y.title = process_axis_label(value_column, units=value_units)
    return standardise_chart(chart)


def magnitude_growth(
    magnitude_growth: pd.DataFrame,
    magnitude_label: str,
    growth_label: str = "Growth",
    #     text_label: str = None,
):
    """"""
    return (
        alt.Chart(
            (
                magnitude_growth
                # Convert growth to fractions
                .assign(growth_=lambda x: x.growth / 100)
                # Reset industries index to a column
                .reset_index()
            ),
            width=300,
            height=300,
        )
        .mark_circle(size=80, color=NESTA_COLOURS[0])
        .encode(
            x=alt.X("magnitude", title=magnitude_label),
            y=alt.Y("growth_", title=growth_label, axis=alt.Axis(format="%")),
            tooltip=["index", "magnitude", "growth"],
        )
        .configure_axis(
            gridDash=[0],
            gridColor="white",
        )
    ).interactive()


def cb_deals_per_year(
    companies: pd.DataFrame,
    funding_rounds: pd.DataFrame,
    company_industries: pd.DataFrame,
) -> ChartType:
    """
    Interactive plot with all the deals (with funding amount info)
    per given year, and extra information about the companies
    """
    df_plot = (
        # Add company data to the investment deal data
        funding_rounds[
            ["org_id", "investment_type", "year", "time_period", "raised_amount_gbp"]
        ]
        .copy()
        .merge(
            companies[
                ["id", "name", "country", "short_description", "long_description"]
            ],
            left_on="org_id",
            right_on="id",
            how="left",
        )
        .drop("id", axis=1)
        # Add info about company industries
        .merge(
            company_industries[["id", "industry"]],
            left_on="org_id",
            right_on="id",
            how="left",
        )
        .drop("id", axis=1)
        # Keep only deals with known raised amounts
        .dropna(subset=["raised_amount_gbp"])
        .query("raised_amount_gbp > 0")
    )

    return (
        alt.Chart(df_plot, width=800, height=400)
        .mark_point(opacity=0.8, size=20, clip=False, color="#6295c4")
        .encode(
            alt.X("year:O"),
            alt.Y("raised_amount_gbp:Q", scale=alt.Scale(type="log")),
            tooltip=[
                "time_period",
                "name",
                "country",
                "short_description",
                "long_description",
                "investment_type",
                "industry",
                "raised_amount_gbp",
            ],
        )
    )
