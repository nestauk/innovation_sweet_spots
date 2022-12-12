""" Utils for making magnitude vs growth trends charts """

import altair as alt
import pandas as pd
from innovation_sweet_spots.utils import plotting_utils as pu

## Parameters: These will stay the same for most figures

# Line between positive and negative growth (usually at 0% growth)
_zero_point = 0
# Opacity of the gradient
_opacity = 0.8
# Gradient colours
_color_emerging = "#FFD47F"
_color_hot = "#F56688"
_color_dormant = "#E4DED9"
_color_stabilising = "#94D8E4"
_epsilon = 0.1
# Fonts
_font_size = 15
_values_format =  ",.3f"
# Axes fine tuning
_tickCountX = 5
_tickCountY = 5
_circle_size = 30
# Trends labels
_trend_text_color = "#524940"
_trend_text_opacity = 0.8
_trend_font_size = _font_size
_dormant_descr = 'Dormant: Relatively low magnitude and low growth'
_emerging_descr = 'Emerging: Relatively low magnitude but high growth'
_hot_descr = 'Hot: Relatively high magnitude and high growth'
_stabilising_descr = 'Stabilising: Relatively high magnitude but low growth'
_trend_descriptions = [_dormant_descr, _emerging_descr, _hot_descr, _stabilising_descr]
# Dummy table
_dummy_data = pd.DataFrame(
    data={
        "magnitude": [0.1, 1, 2, 1.8],
        "growth": [2, 0.5, -0.25, 1.5],
        "category": ["Something", "Anything", "Nothing", "Hot stuff"],
    }
)

## Functions
def test_figure():
    # Figure parameters: These will be unique for each figure
    # Horizontal axis limit
    x_limit = 3
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

    return mangitude_vs_growth_chart(
        _dummy_data,
        x_limit,
        y_limit,
        mid_point,
        baseline_growth,
        values_label,
        text_column,
    )


def gradient_background(
    x_limit: float, y_limit: float, mid_point: int = 1, zero_point: float = _zero_point, x_min: float = 0
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
        x_min,
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
        x_min,
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
    x_min: float = 0,
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

    alpha = mid_point / x_limit
    offset_1 = alpha - _epsilon
    offset_2 = alpha + _epsilon

    return (
        alt.Chart(data)
        .mark_area(
            line={"size": 0},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color=color_start, offset=0),
                    alt.GradientStop(color=color_start, offset=offset_1),
                    alt.GradientStop(color=color_end, offset=offset_2),
                    alt.GradientStop(color=color_end, offset=1),
                ],
                x1=0,
                x2=1,
                y1=y_values,
                y2=y_values,
            ),
            opacity=opacity,
        )
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=(x_min, x_limit)),
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
    baseline_growth: float = None,
    horizontal_log: bool = False,
    width: int = 400,
    height: int = 400,
    font_size: int = _font_size,
    x_min: float = 0,
    values_format: str = _values_format
):
    """Scatter plot component of the magnitude versus growth plot"""
    scale = "log" if horizontal_log else "linear"
    
    fig_points = (
        alt.Chart(data, width=width, height=height)
        .mark_circle(color="black", size=_circle_size, clip=True)
        .encode(
            x=alt.X(
                "magnitude:Q",
                title=horizontal_values_title,
                axis=alt.Axis(tickCount=_tickCountX, labelFlush=False),
                scale=alt.Scale(domain=(x_min, x_limit), type=scale),
            ),
            y=alt.Y(
                "growth:Q",
                title="Growth",
                axis=alt.Axis(format="%", tickCount=_tickCountY, labelFlush=False),
                scale=alt.Scale(domain=(-1, y_limit)),
            ),
            tooltip=[
                alt.Tooltip(f"{text_column}"),                
                alt.Tooltip("magnitude:Q", title=horizontal_values_title, format=_values_format),
                alt.Tooltip("growth:Q", title="Growth", format=".1%"),
            ],
        )
    )

    baseline_rule = (
        alt.Chart(pd.DataFrame({"y": [baseline_growth]}))
        .mark_rule(strokeDash=[5, 7], size=1, color="k")
        .encode(
            y=alt.Y(
                "y:Q",
            )
        )
    )

    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[1, 1], size=1, color="k")
        .encode(
            y=alt.Y(
                "y:Q",
            )
        )
    )

    text = fig_points.mark_text(
        align="left",
        baseline="middle",
        font=pu.FONT,
        dx=7,
        fontSize=_font_size,
        clip=True,
    ).encode(text=f"{text_column}:N")

    if baseline_growth is not None:
        return fig_points + baseline_rule + zero_rule + text
    else:
        return fig_points + zero_rule + text


def configure_trends_chart(fig):
    return fig.configure_axis(
        grid=False,
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL + 2,
    ).configure_view(strokeWidth=0)


def mangitude_vs_growth_chart(
    data: pd.DataFrame,
    x_limit: float,
    y_limit: float,
    mid_point: float,
    baseline_growth: float,
    values_label: str,
    text_column: str,
    horizontal_log: bool = False,
    width: int = 400,
    height: int = 400,
    show_trend_labels: bool = True,
    x_min: float = 0,
):
    """Combines gradient and scatter plots"""
    gradient_bg = gradient_background(x_limit, y_limit, mid_point, x_min=x_min)
    trends_labels_chart = trends_labels(x_limit, y_limit, x_min=x_min)
    scatter = scatter_chart(
        data,
        x_limit,
        y_limit,
        values_label,
        text_column,
        baseline_growth,
        horizontal_log,
        width,
        height,
        x_min = x_min,
    )
    if show_trend_labels:
        return configure_trends_chart(gradient_bg + scatter + trends_labels_chart)
    else:
        return configure_trends_chart(gradient_bg + scatter)
        

def trends_labels(
    x_limit: float,
    y_limit: float,
    text_color = _trend_text_color,
    text_opacity = _trend_text_opacity,
    font_size=_trend_font_size,
    x_min: float=0,
):
    """ Add labels to the plot """
    data = pd.DataFrame(data={
        'label': ['DORMANT', 'EMERGING', 'HOT', 'STABILISING'],        
        'x': [0, 0, x_limit, x_limit],
        'y': [-1, y_limit, y_limit, -1],
        'Trend': _trend_descriptions,
    })
    tooltip = ['Trend']
    
    text_dormant = (
        alt.Chart(data.query("label == 'DORMANT'"))
        .mark_point()
        .encode(
            x='x',
            y='y',
            tooltip=tooltip,
        )
    ).mark_text(
        align="left",
        baseline="bottom",
        font=pu.FONT,
        dx=5,
        fontSize=font_size,
        fontStyle="bold",
        color=text_color,
        opacity=text_opacity,
    ).encode(
        text="label:N")
    
    text_emerging = (
        alt.Chart(data.query("label == 'EMERGING'"))
        .mark_point()
        .encode(
            x='x',
            y='y',
            tooltip=tooltip,
        )
    ).mark_text(
        align="left",
        baseline="top",
        font=pu.FONT,
        dx=5,
        dy=5,          
        fontSize=font_size,
        fontStyle="bold",
        color=text_color, 
        opacity=text_opacity,        
    ).encode(
        text="label:N")

    text_hot = (
        alt.Chart(data.query("label == 'HOT'"))
        .mark_point()
        .encode(
            x='x',
            y='y',
            tooltip=tooltip,            
        )
    ).mark_text(
        align="right",
        baseline="top",
        font=pu.FONT,
        dx=-5,
        dy=5,        
        fontSize=font_size,
        fontStyle="bold",
        color=text_color, 
        opacity=text_opacity,        
    ).encode(
        text="label:N")   
    
    text_stabilising = (
        alt.Chart(data.query("label == 'STABILISING'"))
        .mark_point()
        .encode(
            x='x',
            y='y',
            tooltip=tooltip,            
        )
    ).mark_text(
        align="right",
        baseline="bottom",
        font=pu.FONT,
        dx=-5,
        fontSize=font_size,
        fontStyle="bold",
        color=text_color, 
        opacity=text_opacity,        
    ).encode(
        text="label:N")    
    
    return text_dormant + text_emerging + text_hot + text_stabilising

def _estimate_trend_type(magnitude, growth, mid_point, tolerance=0.1):
    # Flags to double check trend type if ambiguous
    flag_1 = "*" if (abs(magnitude - mid_point) / mid_point) < tolerance else ""
    flag_2 = "*" if abs(growth) < tolerance else ""
    flags = flag_1 + flag_2

    if (growth > 0) and (magnitude < mid_point):
        return "emerging" + flags
    if (growth > 0) and (magnitude >= mid_point):
        return "hot" + flags
    if (growth < 0) and (magnitude < mid_point):
        return "dormant" + flags
    if (growth < 0) and (magnitude >= mid_point):
        return "stable" + flags
    else:
        return "n/a"


def estimate_trend_type(
    mangitude_vs_growth: pd.DataFrame,
    mid_point=None,
    magnitude_column="Magnitude",
    growth_column="growth",
    tolerance=0.1,
):
    """Suggests the trend type provided the magnitude and growth values"""
    mid_point = mangitude_vs_growth[magnitude_column].median()
    trend_types = []
    for i, row in mangitude_vs_growth.iterrows():
        trend_types.append(
            _estimate_trend_type(
                row[magnitude_column], row[growth_column], mid_point, tolerance
            )
        )
    return mangitude_vs_growth.copy().assign(trend_type_suggestion=trend_types)
