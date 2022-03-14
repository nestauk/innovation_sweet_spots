from innovation_sweet_spots import PROJECT_DIR
from jacc_hammer.name_clean import preproc_names
from pathlib import Path
from tempfile import TemporaryDirectory
from jacc_hammer.fuzzy_hash import Cos_config, Fuzzy_config, match_names_stream
import pandas as pd
from utils import merge_matches_with_cb_and_gtr
from typing import List
import typer


STOPWORDS = ["ltd", "llp", "limited", "holdings", "group", "cic", "uk", "plc"]
CB_PATH = PROJECT_DIR / "inputs/data/cb/crunchbase_organizations.csv"
GTR_PATH = PROJECT_DIR / "inputs/data/gtr/gtr_organisations.csv"
CB_GTR_LINK_SAVE_PATH = PROJECT_DIR / "inputs/data/cb_gtr_link"


def link_cb_to_gtr(
    chunksize: int = 100_000,
    sim_mean_min: int = 92,
    stopwords: List[str] = STOPWORDS,
):
    """Link crunchbase organisation ids to gateway to research organisation ids
    using fuzzy matching on the organisation names.

    Saves:

    Lookup csv in the format cb org id -> list of gtr org ids

    Csv which can be used to check the fuzzy matches, containing columns for:
        - cb_org_id
        - cb_address
        - cb_name
        - gtr_org_id
        - gtr_name
        - gtr_address
        - x (index of 'x' list fuzzy matched)
        - y (index of 'y' list fuzzy matched)
        - sim_ratio (levenshtein similarity, 0 to 100)
        - sim_jacc (exact jaccard similarity of the 3-shingles of the names, 0 to 100)
        - sim_cos (tfidf cosine similarity, 0 to 100)
        - sim_mean (mean of sim_ratio, sim_jacc, and sim_cos)

    Args:
        chunksize: Number of chunks to process at a time. Lower the number
            if running out of memory. Defaults to 100_000.
        sim_mean_min: Keep fuzzy matches where the similary mean is above this
            value. Defaults to 92.
    """
    # Load csvs as dataframes
    cb = (
        pd.read_csv(
            CB_PATH,
            index_col=0,
            usecols=["id", "address", "country_code", "name", "legal_name"],
        )
        .query("country_code == 'GBR'")
        .reset_index()
    )
    gtr = pd.read_csv(GTR_PATH)

    # Preprocess names to fuzzy match and convert to lists
    cb_names = preproc_names(cb["name"], stopwords=stopwords).to_list()
    cb_legal_names = preproc_names(cb["legal_name"], stopwords=stopwords).to_list()
    gtr_names = preproc_names(gtr["name"], stopwords=stopwords).to_list()

    # Load configs
    cos_config = Cos_config()
    fuzzy_config = Fuzzy_config()

    # Create temp directory
    tmp_dir = Path(TemporaryDirectory().name)
    tmp_dir.mkdir()

    # Fuzzy match crunchbase org names with gtr org names
    name_matches = pd.concat(
        match_names_stream(
            [cb_names, gtr_names],
            chunksize=chunksize,
            tmp_dir=tmp_dir,
            cos_config=cos_config,
            fuzzy_config=fuzzy_config,
        )
    ).query(f"sim_mean >= {sim_mean_min}")

    # Merge cb and gtr ids to fuzzy matches
    name_matches_cb_gtr = merge_matches_with_cb_and_gtr(
        matches=name_matches, name="name", cb=cb, gtr=gtr
    )

    # Fuzzy match crunchbase org legal names with gtr org names
    legal_name_matches = pd.concat(
        match_names_stream(
            [cb_legal_names, gtr_names],
            chunksize=chunksize,
            tmp_dir=tmp_dir,
            cos_config=cos_config,
            fuzzy_config=fuzzy_config,
        )
    ).query(f"sim_mean >= {sim_mean_min}")

    # Merge cb and gtr ids to fuzzy matches
    legal_name_matches_cb_gtr = merge_matches_with_cb_and_gtr(
        matches=legal_name_matches, name="legal_name", cb=cb, gtr=gtr
    )

    # Concat cb org name matches with cb org legal name matches
    cb_gtr_id_names_and_addresses = pd.concat(
        [name_matches_cb_gtr, legal_name_matches_cb_gtr]
    ).drop_duplicates(subset=["cb_org_id", "gtr_org_id"])
    # Save cb org name, gtr name, ids, and address info csv
    CB_GTR_LINK_SAVE_PATH.mkdir(exist_ok=True)
    cb_gtr_id_names_and_addresses.to_csv(
        CB_GTR_LINK_SAVE_PATH / "cb_gtr_id_names_and_addresses_m.csv"
    )

    # Create cb org id -> gtr org id lookup
    cb_gtr_id_lookup = (
        cb_gtr_id_names_and_addresses.groupby("cb_org_id")["gtr_org_id"]
        .apply(list)
        .reset_index(name="gtr_org_ids")
    )
    # Save cb org id -> gtr org id lookup csv
    cb_gtr_id_lookup.to_csv(CB_GTR_LINK_SAVE_PATH / "cb_gtr_id_lookup_m.csv")


if __name__ == "__main__":
    typer.run(link_cb_to_gtr)
