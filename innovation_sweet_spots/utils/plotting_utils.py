"""
innovation_sweet_spots.utils.plotting_utils

Functions for generating graphs
"""
import altair as alt
import pandas as pd

ChartType = alt.vegalite.v4.api.Chart


def test_chart():
    """Generates a simple test chart"""
    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [1, 2, 3],
                    "y": [10, 15, 30],
                    "label": ["point_1", "point_2", "point_3"],
                }
            )
        )
        .mark_line()
        .encode(alt.X("x"), alt.Y("y"), tooltip=["label"])
    ).interactive()


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
        .mark_line(point=False, stroke="#3e0c59", strokeWidth=1.5)
        .encode(alt.X(f"{x_column}:O"), alt.Y(f"{y_column}:Q"))
    )
    chart.encoding.x.title = process_axis_label(x_column)
    chart.encoding.y.title = process_axis_label(y_column, units=y_units)
    if show_trend:
        chart_trend = chart.transform_regression(x, y).mark_line(
            stroke="black", strokeDash=[2, 2], strokeWidth=0.5
        )
        chart = alt.layer(chart, chart_trend)
    return chart


def cb_deal_types(
    funding_rounds: pd.DataFrame, deal_types: list = None, stack: str = None
) -> ChartType:
    """
    Stacked bar chart of the investment deal types per year

    Args:
        funding_rounds: Dataframe with funding round data
        deal_types: List of permitted deal types
        stack: Either None or 'normalize'
    """
    df = (
        funding_rounds
        if deal_types is None
        else funding_rounds.query("investment_type in @deal_types")
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(
                "count(investment_type)",
                title="Number of deals",
                stack=stack,
            ),
            color="investment_type",
        )
    )


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
