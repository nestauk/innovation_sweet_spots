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
# # Nicer magnitude vs growth plots

# %%
import altair as alt
import pandas as pd

# %%
from innovation_sweet_spots.utils import plotting_utils as pu

# %%
### These will stay the same for most figures
# Line between positive and negative growth (usually at 0% growth)
_zero_point = 0
# Opacity of the gradient
_opacity = 0.8
# Gradient colours
_color_emerging = "#FFD47F"
_color_hot = "#F56688"
_color_dormant = "#E4DED9"
_color_stabilising = "#94D8E4"
# Fonts
_font_size = 15
# Axes fine tuning
_tickCountX = 5
_tickCountY = 5


def gradient_background(
    x_limit: float, y_limit: float, mid_point: int = 1, zero_point: float = _zero_point
):
    """Prepares an altair chart with a gradient background"""
    data_bottom = pd.DataFrame(
        data={
            "x": [0, x_limit],
            "y": [-1, -1],
            "y2": [_zero_point, _zero_point],
        }
    )

    data_top = pd.DataFrame(
        data={
            "x": [0, x_limit],
            "y": [zero_point, zero_point],
        }
    )

    gradient_top = gradient_chart(
        data_top,
        x_limit,
        y_limit,
        True,
        _color_emerging,
        _color_hot,
        mid_point,
        _opacity,
    )
    gradient_bottom = gradient_chart(
        data_bottom,
        x_limit,
        y_limit,
        False,
        _color_dormant,
        _color_stabilising,
        mid_point,
        _opacity,
    )
    return gradient_top + gradient_bottom


def gradient_chart(
    data: pd.DataFrame,
    x_limit: float,
    y_limit: float,
    top: bool = True,
    color_start: str = "green",
    color_end: str = "red",
    mid_point: int = 1,
    opacity: float = 1,
):
    """Creates an altair chart with a gradient block"""
    if top:
        # Top: Gradient block ranges from 0% growth to max value
        y_values = 0
        y2 = alt.value(0)
    else:
        # Bottom: Gradient block ranges from -100% growth to 0%
        y_values = 0
        y2 = "y2"
    return (
        alt.Chart(data)
        .mark_area(
            line={"size": 0},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color=color_start, offset=0),
                    alt.GradientStop(color=color_end, offset=1),
                ],
                x1=0,
                x2=mid_point / x_limit,
                y1=y_values,
                y2=y_values,
            ),
            opacity=opacity,
        )
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=(0, x_limit)),
                axis=alt.Axis(tickCount=_tickCountX),
                title="",
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=(-1, y_limit)),
                axis=alt.Axis(format="%", tickCount=_tickCountY),
                title="",
            ),
            y2=y2,
        )
    )


def scatter_chart(
    data: pd.DataFrame,
    x_limit: float,
    y_limit: float,
    horizontal_values_title: str,
    text_column: str,
    width: int = 400,
    height: int = 400,
    font_size: int = _font_size,
):
    """Scatter plot component of the magnitude versus growth plot"""
    fig_points = (
        alt.Chart(data, width=width, height=height)
        .mark_circle(color="black", size=30)
        .encode(
            x=alt.X(
                "magnitude:Q",
                title=horizontal_values_title,
                axis=alt.Axis(tickCount=_tickCountX, labelFlush=False),
                scale=alt.Scale(domain=(0, x_limit)),
            ),
            y=alt.Y(
                "growth:Q",
                title="Growth",
                axis=alt.Axis(format="%", tickCount=_tickCountY, labelFlush=False),
                scale=alt.Scale(domain=(-1, y_limit)),
            ),
            tooltip=[
                alt.Tooltip("magnitude:Q", title=horizontal_values_title),
                alt.Tooltip("growth:Q", title="Growth", format=".1%"),
            ],
        )
    )

    baseline_rule = (
        alt.Chart(pd.DataFrame({"y": [baseline_growth]}))
        .mark_rule(strokeDash=[5, 7], size=1, color="k")
        .encode(y="y:Q")
    )

    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[1, 1], size=1, color="k")
        .encode(y="y:Q")
    )

    text = fig_points.mark_text(
        align="left",
        baseline="middle",
        font=pu.FONT,
        dx=7,
        fontSize=_font_size,
    ).encode(text=f"{text_column}:N")

    return fig_points + baseline_rule + zero_rule + text


def configure_trends_chart(fig):
    return fig.configure_axis(
        grid=False,
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    ).configure_view(strokeWidth=0)


def mangitude_vs_growth_chart(
    data: pd.DataFrame,
    x_limit: float,
    y_limit: float,
    mid_point: float,
    baseline_growth: float,
    values_label: str,
    text_column: str,
    width: int = 400,
    height: int = 400,
):
    """Combines gradient and scatter plots"""
    gradient_bg = gradient_background(x_limit, y_limit, mid_point)
    scatter = scatter_chart(
        data, x_limit, y_limit, values_label, text_column, width, height
    )
    return configure_trends_chart(gradient_bg + scatter)


# %%
# These will be unique for each figure
# Horizontal axis limit
x_limit = 2
# Vertical axis limit
y_limit = 4
# Gradient's horizontal mid-point (usually median value)
mid_point = 1
# Baseline growth
baseline_growth = 0.1
# Horizontal values title
values_label = "Dummy funding (millions GBP)"
# Column to use for data point labels
text_column = "category"

# Dummy table
dummy_data = pd.DataFrame(
    data={
        "magnitude": [0.1, 1, 2, 1.8],
        "growth": [2, 0.5, -0.25, 1.5],
        "category": ["Something", "Anything", "Nothing", "Hot stuff"],
    }
)


# To Do: Center the labels
mangitude_vs_growth_chart(
    dummy_data,
    3,
    y_limit,
    mid_point,
    baseline_growth,
    values_label,
    text_column,
)


# %%
# Test the refactored utils
from innovation_sweet_spots.utils import chart_trends
import importlib

importlib.reload(chart_trends)

# %%
chart_trends.mangitude_vs_growth_chart(
    chart_trends._dummy_data,
    3,
    3,
    1,
    0.25,
    "Dummy data",
    "category",
)

# %%
chart_trends.test_figure()

# %%
