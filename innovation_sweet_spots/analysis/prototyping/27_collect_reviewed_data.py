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
import pandas as pd
import numpy as np

import innovation_sweet_spots.utils.io as iss_io
from innovation_sweet_spots import PROJECT_DIR

# %%
DF_REF = iss_io.load_pickle(
    PROJECT_DIR / "outputs/data/results_august/reference_category_data_2.p"
)

# %%
list(DF_REF.keys())

# %%
# DF_REF['Solar']

# %%

# %%
