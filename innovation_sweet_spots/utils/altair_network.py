# Plot network data using altair
import pandas as pd
import altair as alt
import networkx as nx


def node_layer(
    node_df,
    graph,
    node_label_lookup,
    node_label,
    node_size,
    node_color,
    node_opacity,
    show_neighbours,
    **kwargs
):
    """Creates node_layer in the plot"""
    node_chart = (
        alt.Chart(node_df)
        .mark_point(filled=True, stroke="black", strokeWidth=0.5)
        .encode(x=alt.X("x", axis=None), y=alt.Y("y", axis=None))
    )

    if node_size in node_df.columns:
        node_chart = node_chart.encode(
            size=alt.Size(
                node_size,
                title=kwargs["node_size_title"],
                legend=alt.Legend(orient="bottom"),
            )
        )

    if node_color in node_df.columns:
        # node_chart = node_chart.encode(
        #     color=alt.condition(f'datum.{node_color} !== null',
        #                         alt.Color(
        #                                   node_color,
        #                                   title=kwargs["node_color_title"],
        #                                   legend=alt.Legend(columns=2),
        #                                   scale=alt.Scale(scheme="tableau20"),
        #                                   sort="descending"),
        #                         alt.value('lightgray')))
        node_chart = node_chart.encode(
            color=alt.Color(
                node_color,
                title=kwargs["node_color_title"],
                legend=alt.Legend(columns=2),
                scale=alt.Scale(scheme="tableau20"),
                sort="descending",
            )
        )

    if node_opacity in node_df.columns:

        node_chart = node_chart.encode(
            opacity=alt.Opacity(node_opacity, title=kwargs["node_opacity_title"])
        )

    if show_neighbours is True:
        neighbors = {
            node: ", ".join(
                [str(node_label_lookup[n]) for n in nx.neighbors(graph, node)]
            )
            for node in graph.nodes()
        }
        node_df["neighbors"] = node_df["node"].map(neighbors)

        node_chart = node_chart.encode(tooltip=[node_label, "neighbors"])
    else:
        node_chart = node_chart.encode(tooltip=[node_label])

    return node_chart


def calculate_edge_positions(graph, node_pos_lookup, edge_scale):
    """Calculates the positions of the edges"""
    if len(nx.get_edge_attributes(graph, "weight")) > 0:
        weighted = True
        edges_df = pd.DataFrame(
            [
                {"e1": e[0], "e2": e[1], "weight": e[2]["weight"]}
                for e in list(graph.edges(data=True))
            ]
        )
    else:
        weighted = False
        edges_df = pd.DataFrame(
            [
                {"e1": e[0], "e2": e[1], "weight": 1}
                for e in list(graph.edges(data=True))
            ]
        )

    edge_pos_cont = []

    for _id, r in edges_df.iterrows():
        x1, y1 = [node_pos_lookup[r["e1"]][n] for n in [0, 1]]
        x2, y2 = [node_pos_lookup[r["e2"]][n] for n in [0, 1]]
        w = int(r["weight"]) / edge_scale
        edge_pos_cont.append(
            pd.Series([x1, y1, x2, y2, w], index=["x1", "y1", "x2", "y2", "weight"])
        )

    edges_pos_df = pd.DataFrame(edge_pos_cont)
    return edges_pos_df, weighted


def edge_layer(edges_pos_df, weighted, edge_opacity, **kwargs):
    """Creates edge layer in the plot"""
    edge_chart = (
        alt.Chart(edges_pos_df)
        .mark_line()
        .encode(x="x1", x2="x2", y="y1", y2="y2", strokeOpacity=alt.value(edge_opacity))
    )
    if weighted is True:
        edge_chart = edge_chart.encode(
            strokeWidth=alt.StrokeWidth(
                "weight",
                title=kwargs["edge_weight_title"],
                legend=alt.Legend(orient="bottom"),
            )
        )
    return edge_chart


def plot_altair_network(
    node_df,
    graph,
    node_label=None,
    node_size=None,
    node_color=None,
    node_opacity=None,
    show_neighbours=True,
    edge_scale=1,
    edge_opacity=0.1,
    **kwargs
):
    """Plot a network graph with altair
    Args:
        node_df (df): dataframe where the rows are nodes and the
        node_label (str): node label variable in node_df
        node_size (str): node size variable in node_df
        node_color (str): node color variable (this is a categorical variable)
        columns are relevant variables including node position, node_label, node_size,node_color
        graph (networkx graph): graph object generated with networkx that we use to extract edges & edgeWidths
        show_neighbours (bool): if we want neighbours to be extracted and showed in a tooltip
        edge_scale (float): scale for weight value
        edge_opacity (float): weight opacity
    """

    # Node chart
    node_df_ = node_df.copy()

    # Make node name - label lookup
    node_label_lookup = node_df_.set_index("node")["node_name"].to_dict()
    node_pos_lookup = {r["node"]: (r["x"], r["y"]) for _id, r in node_df_.iterrows()}

    # Plot nodes
    node_plot = node_layer(
        node_df_,
        graph,
        node_label_lookup,
        node_label,
        node_size,
        node_color,
        node_opacity,
        show_neighbours,
        **kwargs
    )
    # Plot edges (after calculating their positions)
    edge_positions, weighted = calculate_edge_positions(
        graph, node_pos_lookup, edge_scale
    )

    edge_plot = edge_layer(edge_positions, weighted, edge_opacity, **kwargs)

    # Combine plots

    net_plot = (
        (node_plot + edge_plot).properties(title=kwargs["title"])
        # .configure_axis(grid=False)
        # .configure_view(strokeWidth=0)
    )
    return net_plot
