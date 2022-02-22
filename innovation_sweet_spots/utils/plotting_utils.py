"""
innovation_sweet_spots.utils.plotting_utils

"""
import altair as alt
import pandas as pd


def process_axis_label(text: str, units: str = None):
    """Turns snake case into normal text and capitalises first letter"""
    new_text = " ".join(text.split("_")).capitalize()
    if units is not None:
        new_text += f" ({units})"
    return new_text


def time_series(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_units: str = None,
    show_trend: bool = False,
) -> alt.Chart:
    chart = (
        alt.Chart(data, width=400, height=200)
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
