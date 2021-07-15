from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import load_pickle


def get_green_gtr_docs(
    fpath=PROJECT_DIR / "outputs/data/gtr/gtr_green_docs_tokenised.p",
):
    return load_pickle(fpath)


def get_green_cb_docs(fpath=PROJECT_DIR / "outputs/data/cb/cb_green_docs_tokenised.p"):
    return load_pickle(fpath)
