# Script to train topic model on creative industries projects
import logging

from numpy.random import seed

from createch import PROJECT_DIR
from createch.getters.gtr import (
    get_cis_lookup,
    get_gtr_projects,
    get_gtr_tokenised,
    get_link_table,
    get_organisations,
)
from createch.getters.processing import save_model
from createch.pipeline.topic_modelling import post_process_model_clusters, train_model
from createch.utils.io import save_lookup


def make_creative_tokenised(projects, orgs):
    """Creates lookup between project ids and descriptions for
    # creative industries projects"""

    logging.info("Merging tables")
    link_table = get_link_table().query("table_name == 'gtr_organisations'")

    project_org = projects.merge(
        link_table, left_on="project_id", right_on="project_id"
    ).merge(orgs, left_on="id", right_on="gtr_id")
    ci_projects = set(project_org["project_id"])

    logging.info(len(ci_projects))

    logging.info("Getting tokenised")
    ci_tokenised = {k: v for k, v in get_gtr_tokenised().items() if k in ci_projects}
    return ci_tokenised


if __name__ == "__main__":

    seed(999)
    logging.info("Reading data")

    SIC_IND_LOOKUP = get_cis_lookup()
    projects = get_gtr_projects()
    ci_orgs = get_organisations()

    ci_tokenised = make_creative_tokenised(projects, ci_orgs)

    logging.info("Training topic model")
    gtr_topsbm = train_model(list(ci_tokenised.values()), list(ci_tokenised.keys()))

    topic_mix_df, clusters = post_process_model_clusters(
        gtr_topsbm, top_level=0, cl_level=1
    )
    doc_to_cluster_lookup = {doc[0]: k for k, v in clusters.items() for doc in v}

    save_model(gtr_topsbm, "outputs/models/gtr/gtr_topsbm_creative")
    topic_mix_df.to_csv(f"{PROJECT_DIR}/outputs/data/gtr/gtr_topic_mix.csv")
    save_lookup(doc_to_cluster_lookup, "outputs/data/gtr/project_cluster_lookup")
