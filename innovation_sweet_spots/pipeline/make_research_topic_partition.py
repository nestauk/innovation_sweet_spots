# Build a network of topic co-occurrences in projects
# and decompose into 'discipline communities'

import logging

import graph_tool.all as gt
import networkx as nx
from numpy.random import seed

from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.pipeline.fetch_daps1_data.daps1_utils import get_daps_table
from innovation_sweet_spots.pipeline.network_analysis import make_network_from_coocc
from innovation_sweet_spots.utils.io import save_lookup

GTR_DATA_PATH = f"{PROJECT_DIR}/inputs/data/gtr"


def make_network_analysis_inputs() -> list:
    """Reads relevant tables"""
    gtr_projects = get_daps_table("gtr_projects", GTR_DATA_PATH)[
        ["project_id", "abstractText", "leadFunder"]
    ]
    gtr_link = (
        get_daps_table("gtr_link_table", GTR_DATA_PATH)
        .query("table_name=='gtr_topic'")
        .reset_index(drop=True)
    )
    gtr_categories = get_daps_table("gtr_topics", GTR_DATA_PATH)

    categories_projects = (
        gtr_categories.merge(gtr_link, on="id")
        .query("topic_type=='researchTopic'")
        .query("text!='Unclassified'")[["project_id", "text"]]
        .reset_index(drop=True)
    )

    return gtr_projects, categories_projects


def make_gt_network(net: nx.Graph) -> list:
    """Converts co-occurrence network to graph-tool network"""
    nodes = {name: n for n, name in enumerate(net.nodes())}
    index_to_name = {v: k for k, v in nodes.items()}
    edges = list(net.edges(data=True))

    g_net = gt.Graph(directed=False)
    g_net.add_vertex(len(net.nodes))

    eprop = g_net.new_edge_property("int")
    g_net.edge_properties["weight"] = eprop

    for edg in edges:
        n1 = nodes[edg[0]]
        n2 = nodes[edg[1]]

        e = g_net.add_edge(g_net.vertex(n1), g_net.vertex(n2))
        g_net.ep["weight"][e] = edg[2]["weight"]

    return g_net, index_to_name


def get_community_names(partition, index_to_name, level=1):
    """Create node - community lookup"""

    b = partition.get_bs()

    b_lookup = {n: b[level][n] for n in sorted(set(b[0]))}

    names = {index_to_name[n]: int(b_lookup[c]) for n, c in enumerate(b[0])}

    return names


if __name__ == "__main__":
    seed(888)

    logging.info("Make inputs")
    gtr_projects, categories_projects = make_network_analysis_inputs()

    logging.info("Build co-occurrence network")
    cat_coocc = categories_projects.groupby("project_id")["text"].apply(
        lambda x: list(x)
    )

    net = make_network_from_coocc(cat_coocc, spanning=False)
    gt_net, index_to_name = make_gt_network(net)

    logging.info("Extract communities")
    state = gt.minimize_nested_blockmodel_dl(gt_net)

    comm_names = get_community_names(state, index_to_name)
    save_lookup(comm_names, "outputs/data/gtr/topic_community_lookup")
