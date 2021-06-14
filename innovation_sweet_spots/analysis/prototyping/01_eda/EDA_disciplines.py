# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR, config
from innovation_sweet_spots.utils.io import get_lookup
import pandas as pd

# %%
comms = get_lookup("outputs/data/gtr/topic_community_lookup")

# %%
discipline_names = config["discipline_classification"]["discipline_names"]
discipline_names_lookup = dict(zip(range(0, len(discipline_names)), discipline_names))


# %%
topics_communities = pd.DataFrame(
    list(zip(comms.keys(), comms.values())), columns=["research_topic", "community"]
)
topics_communities["discipline"] = topics_communities.community.apply(
    lambda x: discipline_names_lookup[x]
)


# %%
topics_communities.query("community==0")

# %%
