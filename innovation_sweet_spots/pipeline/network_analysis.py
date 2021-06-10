from itertools import chain, combinations

import networkx as nx
import pandas as pd


def make_topic_coocc(topic_mix, thres):

    co_occ = (
        topic_mix.reset_index(drop=False)
        .melt(id_vars="index")
        .query(f"value>{thres}")
        .reset_index(drop=False)
        .groupby("index")["variable"]
        .apply(lambda x: list(x))
    )
    return co_occ


def make_network_from_coocc(
    co_occ: list, thres: float = 0.1, extra_links: int = 200, spanning: bool = True
) -> nx.Graph:
    """Create a network from a list of co-occurring terms
    Args
        co_occ: each element is a list of co-occurring entities
        thres: maximum occurrence rate
        weight_thres: extra edges to add
        spanning: filter the network with a maximum spanning tree
    """

    # Make weighted edge list
    pairs = list(chain(*[sorted(list(combinations(x, 2))) for x in co_occ]))
    pairs = [x for x in pairs if len(x) > 0]

    edge_list = pd.DataFrame(pairs, columns=["source", "target"])

    edge_list["weight"] = 1

    edge_list_weighted = (
        edge_list.groupby(["source", "target"])["weight"].sum().reset_index(drop=False)
    )

    # Make and post-process network
    net = nx.from_pandas_edgelist(edge_list_weighted, edge_attr=True)

    if spanning is True:
        msp = nx.maximum_spanning_tree(net)
        msp_plus = make_msp_plus(net, msp, thres=extra_links)
        return msp_plus

    else:
        return net


def make_msp_plus(net: nx.Graph, msp: nx.Graph, thres: int = 200) -> nx.Graph:
    """Create a network combining maximum spanning tree and top edges
    Args:
        net: original network
        msp: maximum spanning tree of the original network
        thres: extra edges to aadd
    Returns:
        A network
    """

    msp_ed = set(msp.edges())

    top_edges_net = nx.Graph(
        [
            x
            for x in sorted(
                net.edges(data=True),
                key=lambda x: x[2]["weight"],
                reverse=True,
            )
            if (x[0], x[1]) not in msp_ed
        ][:thres]
    )

    # Combines them
    united_graph = nx.Graph(
        list(msp.edges(data=True)) + list(top_edges_net.edges(data=True))
    )
    return united_graph
