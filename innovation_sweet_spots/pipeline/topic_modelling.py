import logging

import pandas as pd
from innovation_sweet_spots.hSBM_Topicmodel.sbmtm import sbmtm
from innovation_sweet_spots import logging


def train_model(corpus, doc_ids):
    """Trains top sbm model on tokenised corpus"""
    model = sbmtm()
    model.make_graph(corpus, documents=doc_ids)
    logging.info(f"Fitting a topic model using {len(corpus)} documents")
    model.fit()
    logging.info(f"Topic model ready")
    return model


def post_process_model(model, top_level, top_words=5):
    """Function to post-process the outputs of a hierarchical topic model
    _____
    Args:
      model: A hsbm topic model
      top_level: The level of resolution at which we want to extract topics
      top_words: top_words to include in the topic name
    _____
    Returns:
      A topic mix df with topics and weights by document
    """
    # Extract the word mix (word components of each topic)
    logging.info("Creating topic names")
    word_mix = model.topics(l=top_level)

    # Tidy names
    topic_name_lookup = {
        key: "_".join([x[0] for x in values[:top_words]])
        for key, values in word_mix.items()
    }
    topic_names = list(topic_name_lookup.values())

    # Extract the topic mix df
    logging.info("Extracting topics")
    topic_mix_ = pd.DataFrame(
        model.get_groups(l=top_level)["p_tw_d"].T,
        columns=topic_names,
        index=model.documents,
    )

    return topic_mix_


def filter_topics(topic_df, presence_thr, prevalence_thr):
    """Filter uninformative ("stop") topics
    Args:
        top_df (df): topics
        presence_thr (int): threshold to detect topic in article
        prevalence_thr (int): threshold to exclude topic from corpus
    Returns:
        Filtered df
    """
    # Remove highly uninformative / generic topics

    topic_prevalence = (
        topic_df.applymap(lambda x: x > presence_thr)
        .mean()
        .sort_values(ascending=False)
    )

    # Filter topics
    filter_topics = topic_prevalence.index[topic_prevalence > prevalence_thr].tolist()

    # We also remove short topics (with less than two ngrams)
    filter_topics = filter_topics + [
        x for x in topic_prevalence.index if len(x.split("_")) <= 2
    ]

    topic_df_filt = topic_df.drop(filter_topics, axis=1)

    return topic_df_filt, filter_topics


def post_process_model_clusters(model, top_level, cl_level, top_thres=1, top_words=5):
    """Function to post-process the outputs of a hierarchical topic model
    _____
    Args:
        model: A hsbm topic model
        top_level: The level of resolution at which we want to extract topics
        cl_level:The level of resolution at which we want to extract clusters
        top_thres: The maximum share of documents where a topic appears.
        1 means that all topics are included
        top_words: number of words to use when naming topics

    _____
    Returns:
      A topic mix df with topics and weights by document
      A lookup between ids and clusters
    """
    # Extract the word mix (word components of each topic)
    topic_mix_ = post_process_model(model, top_level, top_words)

    # word_mix = model.topics(l=top_level)

    # # Create tidier names
    # topic_name_lookup = {
    #     key: "_".join([x[0] for x in values[:5]]) for key, values in word_mix.items()
    # }
    # topic_names = list(topic_name_lookup.values())

    # # Extract the topic mix df
    # topic_mix_ = pd.DataFrame(
    #     model.get_groups(l=top_level)["p_tw_d"].T,
    #     columns=topic_names,
    #     index=model.documents,
    # )

    # Remove highly uninformative / generic topics
    topic_prevalence = (
        topic_mix_.applymap(lambda x: x > 0).mean().sort_values(ascending=False)
    )
    filter_topics = topic_prevalence.index[topic_prevalence < top_thres]
    topic_mix = topic_mix_[filter_topics]

    # Extract the clusters to which different documents belong (we force all documents
    # to belong to a cluster)
    cluster_assignment = model.clusters(l=cl_level, n=len(model.documents))
    # cluster_sets = {
    #     c: set([x[0] for x in papers]) for c, papers in cluster_assigment.items()
    # }

    # # Assign topics to their clusters
    # topic_mix["cluster"] = [
    #     [f"cluster_{n}" for n, v in cluster_sets.items() if x in v][0]
    #     for x in topic_mix.index
    # ]

    return topic_mix, cluster_assignment
