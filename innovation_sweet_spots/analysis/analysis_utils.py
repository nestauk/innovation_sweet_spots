"""
Utils for doing data analysis

"""
from typing import Iterator
import pandas as pd

### Preparation


def create_documents_from_dataframe(
    df: pd.DataFrame, columns: Iterator[str]
) -> Iterator[str]:
    """Build documents from texts in the table columns"""
    # Select columns to include in the document
    df_ = df[columns].fillna("").copy()
    # Preprocess project text
    text_lists = [df_[col].to_list() for col in columns]
    # Create project documents
    docs = [preprocess_text(text) for text in create_documents(text_lists)]
    return docs


def create_documents(lists_of_texts: Iterator[str]) -> Iterator[str]:
    """
    Create documents from lists of texts for further analysis, e.g. to
    calculate tf-idf scores of n-grams. For example:
        (['one','two'], ['cat', 'dogs']) -> ['one cat', 'two dogs']

    Parameters
    ----------
        lists_of_texts:
            Contains lists of texts to be joined up and processed to create
            the "documents"; i-th element of each list corresponds to the
            i-th entity/document

    Yields
    ------
        Iterator[str]
            Document generator
    """
    # Check if all lists have the same length
    if len({len(i) for i in lists_of_texts}) == 1:
        # Transpose the lists of skill texts
        transposed_lists_of_texts = map(list, zip(*lists_of_texts))
        # Join up the skill texts for each skills entity
        return (
            " ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


def preprocess_text(text: str) -> str:
    """Placeholder for some more serious text preprocessing"""
    return text.lower().strip()


### Search term analysis


def is_term_present(search_term: str, docs: Iterator[str]) -> Iterator[bool]:
    """Simple method to check if a keyword or keyphrase is in the set of documents"""
    return [search_term in doc for doc in docs]


def search_in_items(search_term: str, docs: Iterator[str], item_table: pd.DataFrame):
    """Returns table with only the items whose documents contain the search term"""
    bool_mask = is_term_present(search_term, docs)
    return item_table[bool_mask].copy()
