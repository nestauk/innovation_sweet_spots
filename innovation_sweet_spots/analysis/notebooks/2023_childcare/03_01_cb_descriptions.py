# %%
import typer
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler
from innovation_sweet_spots.getters import crunchbase
from innovation_sweet_spots.utils.text_processing_utils import (
    create_documents_from_dataframe,
)
from innovation_sweet_spots.utils import text_cleaning_utils as tcu

from typing import List, Iterable
from toolz import pipe
import re
import spacy
import csv
import pandas as pd


# %%
OUTPUT_FILE = PROJECT_DIR / "outputs/preprocessed/texts/cb_descriptions_formatted.csv"

# %%
NLP = spacy.load("en_core_web_sm")
CB = CrunchbaseWrangler()

# Data parameters
COLUMNS = ["short_description", "long_description", "_industries"]

# %%
def create_industries_string(industries: List, suffix: str = "Industries: ") -> str:
    """Creates a string of industries for a given organisation"""
    # Keep only strings from the input list
    _industries = sorted([i for i in industries if isinstance(i, str)])
    if len(_industries) > 0:
        return suffix + ", ".join(_industries)
    else:
        return ""


def replace_company_names(text: str, company_name: str) -> str:
    """Replaces company names with the word "company" in a given text"""
    return text.replace(company_name, "The company")


def fix_double_puncts(text: str) -> str:
    """Fixes double punctuation in a given text"""
    return text.replace(r"..", ".")


def remove_locations_orgs_and_people_names(text: str, nlp) -> str:
    """Removes locations and people's names from a given text"""
    doc = nlp(text)
    return " ".join(
        [token.text for token in doc if not token.ent_type_ in ["GPE", "PERSON", "ORG"]]
    )


def remove_websites(text: str) -> str:
    """Removes websites from a given text"""
    return re.sub(r"http\S+", "", text)


# use regex to replace all different parenthesis symbols with spaces
def remove_parenthesis(text: str) -> str:
    """Removes parenthesis from a given text"""
    return re.sub(r"\(|\)|\[|\]|\{|\}", " ", text)


def fix_multiple_spaces(text: str) -> str:
    """Fixes multiple spaces in a given text"""
    return re.sub(r"\s+", " ", text)


def remove_sentences_without_text(text: str) -> str:
    """Removes sentences without text from a given text using split"""
    return ". ".join([s for s in text.split(".") if len(s) > 0])


# Replace cases of multiple subsequent punctuation symbols (eg, commas, dots) with a single dot
def remove_multiple_puncts(text: str) -> str:
    """Removes multiple punctuation symbols from a given text"""
    return re.sub(r"(\.|,){2,}", ".", text)


def process_company_description(text: str, company_name: str, nlp) -> str:
    """Processes a company description"""
    return pipe(
        text,
        lambda x: replace_company_names(x, company_name),
        lambda x: remove_websites(x),
        lambda x: remove_locations_orgs_and_people_names(x, nlp),
        lambda x: tcu.unpad_punctuation(x),
        lambda x: remove_parenthesis(x),
        lambda x: remove_multiple_puncts(x),
        lambda x: fix_multiple_spaces(x),
    )


# %%
def process_descriptions(
    nrows: int = None,
    last_index: str = None,
) -> None:

    # Check the last index
    try:
        last_index = (
            len(pd.read_csv(OUTPUT_FILE, names=["id", "name", "description"]))
            if last_index is None
            else last_index
        )
    except FileNotFoundError:
        last_index = -1

    logging.info(f"Getting started, will commence from the last index: {last_index}")
    logging.info(f"Loading data from Crunchbase, nrows: {nrows}")

    # Get the full CB data
    cb_data = crunchbase.get_crunchbase_orgs(nrows=nrows).drop_duplicates("id")
    #  Create a column listing company's industries
    cb_industries = CB.get_company_industries(cb_data, return_lists=True)
    cb_data["_industries"] = cb_industries["industry"].apply(
        lambda x: create_industries_string(x)
    )
    # Unprocessed descriptions
    text_documents_raw = create_documents_from_dataframe(cb_data, columns=COLUMNS)

    # Check the length of the descriptions
    lens = [len(doc) for doc in text_documents_raw]
    # Get the indices of the valid descriptions
    indices_ok = [i for i, l in enumerate(lens) if l >= 75]

    ids = [cb_id for i, cb_id in enumerate(cb_data.id.to_list()) if i in indices_ok]
    names = [name for i, name in enumerate(cb_data.name.to_list()) if i in indices_ok]
    text_documents_raw = [
        text for i, text in enumerate(text_documents_raw) if i in indices_ok
    ]

    with open(OUTPUT_FILE, mode="a") as file:
        writer = csv.writer(file)
        for i, _ in enumerate(indices_ok):
            if i <= last_index:
                continue
            else:
                writer.writerow(
                    [
                        ids[i],
                        names[i],
                        process_company_description(
                            text_documents_raw[i], names[i], NLP
                        ),
                    ]
                )
                # Output the progress every 5000 rows
                if i % 5000 == 0:
                    logging.info(f"Processed {i} rows")


# %%
if __name__ == "__main__":
    typer.run(process_descriptions)
