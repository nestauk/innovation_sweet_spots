"""
innovation_sweet_spots.analysis.lda_modelling_utils

Module for analysing LDA topic modelling results and querying documents using a model
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.utils.io import (
    save_text_items,
    read_text_items,
)

import pandas as pd
import numpy as np
import tomotopy as tp
from tomotopy import LDAModel
import pyLDAvis

from typing import Iterator, Dict
from os import PathLike
from numpy.typing import ArrayLike


def print_model_info(mdl: LDAModel):
    """
    Helper function to output basic info about the LDA model:
        - Num docs: Number of documents used to generate the model
        - Num Vocabs: Number of words (tokens) used to generate the model
        - Total Words: Total number of words (tokens), including unused ones
        - Removed Top words: Frequent words that have been removed from the model
    """
    logging.info(
        "Num docs:{}, Num vocabs:{}, Total words:{}".format(
            len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
        )
    )
    logging.info("Removed top words: ", *mdl.removed_top_words)


def get_topic_token_distributions(mdl: LDAModel) -> ArrayLike:
    """
    Gets model's topics' word (token) distributions.

    Args:
        mdl: Tomotopy topic model (usually LDAModel)

    Returns:
        Numpy array with the shape (n_topics, n_words) where each
        element (i, j) is the probability of the word j belonging
        to the topic i
    """
    return np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])


def get_relevant_topic_tokens(
    mdl: LDAModel, n: int = 10, _lambda: float = 0.5
) -> Dict[int, Iterator[str]]:
    """
    Gets the top n most relevant topic tokens. Relevance is calculated following
    Sievert & Shirley (2014) [https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf]

    Args:
        mdl: Tomotopy topic model (usually LDAModel)
        n: Number of words (tokens) to return for each topic
        _lambda: Relevance calculation parameter in the range [0, 1];
            When _lambda=1, token relevance is equivalent to their topic-
            specific probability, whereas _lambda=0 will rank terms solely
            by their 'lift' (ratio of tokens' topic-specific vs corpus probability).
            Parameter _lambda=0.5 was found to be in the optimal range by Sievert & Shirley (2014).

    Returns:
        Dictionary of the form {topic_number: list of most relevant tokens}

    """
    # Number of documents
    n_documents = len(mdl.docs)
    # Token corpus probabilities
    token_corpus_probabilities = mdl.used_vocab_df / n_documents
    # Model's topics' token distributions
    token_topic_distributions = get_topic_token_distributions(mdl)
    # Iterate through topics and get the most relevant tokens
    most_relevant_tokens = dict()
    for t in range(mdl.k):
        # See p.66 in Sievert & Shirley (2014)
        relevance = _lambda * np.log(token_topic_distributions[t, :]) + (
            1 - _lambda
        ) * np.log(token_topic_distributions[t, :] / token_corpus_probabilities)
        top_tokens = [mdl.used_vocabs[i] for i in np.flip(np.argsort(relevance))[0:n]]
        most_relevant_tokens[t] = top_tokens
    return most_relevant_tokens


def load_lda_model(model_name: str, folder: PathLike) -> LDAModel:
    """Loads a tomotopy LDAModel topic model"""
    filepath = folder / f"{model_name}.bin"
    logging.info(f"Loading a LDA model from {filepath}")
    return tp.LDAModel.load(str(filepath))


def save_lda_model(topic_model: LDAModel, model_name: str, folder: PathLike):
    """Saves a tomotopy topic model"""
    filepath = folder / f"{model_name}.bin"
    logging.info(f"Saving a LDA model to {filepath}")
    topic_model.save(str(filepath))


def load_topic_descriptions(model_name: str, folder: PathLike) -> pd.DataFrame:
    """Loads a table of the model topic descriptions"""
    return pd.read_csv(folder / f"{model_name}_topic_descriptions.csv")


def save_topic_descriptions(
    topic_descriptions: pd.DataFrame, model_name: str, folder: PathLike
) -> pd.DataFrame:
    """Loads a table of the model topic descriptions"""
    return topic_descriptions.to_csv(
        folder / f"{model_name}_topic_descriptions.csv", index=False
    )


def load_document_ids(model_name: str, folder: PathLike) -> Iterator[str]:
    """Loads a list with the document identifiers"""
    return read_text_items(folder / f"{model_name}_document_ids.txt")


def create_topic_description_table(
    topic_model: LDAModel, manual_labels: Iterator[str] = None
):
    """
    Generates a data frame with topic descriptions
    """
    # Get manual labels
    if manual_labels is None:
        manual_labels = [None] * topic_model.k
    else:
        assert (
            len(manual_labels) == topic_model.k
        ), "The number of manual labels does not match the number of topics"
    # Get relevant tokens
    relevant_tokens = get_relevant_topic_tokens(topic_model, n=10, _lambda=0.5)
    # Generate the table
    return pd.DataFrame(
        [
            {
                "token_id": i,
                "manual_label": manual_labels[i],
                "most_relevant_tokens": " ".join(relevant_tokens[i]),
            }
            for i in range(topic_model.k)
        ]
    )


def save_lda_model_data(
    model_name: str,
    folder: PathLike,
    topic_model: LDAModel,
    document_ids: Iterator[str],
    manual_labels: Iterator[str] = None,
):
    """
    Saves the data associated with the topic model:
        - A tomotopy topic model (.bin file)
        - A list of document ids (.txt file)
        - A table with topic descriptions (.csv file)
    """
    save_lda_model(topic_model, model_name, folder)
    save_text_items(document_ids, folder / f"{model_name}_document_ids.txt")
    save_topic_descriptions(
        create_topic_description_table(topic_model, manual_labels), model_name, folder
    )


def load_lda_model_data(model_name: str, folder: PathLike) -> dict:
    """
    Loads files pertaining to a topic model

    Returns:
        Dictionary with the following keys:
            - model: A tomotopy topic model
            - document_ids: A list of document ids
            - topic_descriptions: A table with topic descriptions
    """
    model_dict = {
        "model_name": model_name,
        "model": load_lda_model(model_name, folder),
        "document_ids": load_document_ids(model_name, folder),
        "topic_descriptions": load_topic_descriptions(model_name, folder),
    }
    # Data consistency checks
    assert len(model_dict["document_ids"]) == len(
        model_dict["model"].docs
    ), "The number of documents in the model is different from the number of document identifiers"
    assert (
        len(model_dict["topic_descriptions"]) == model_dict["model"].k
    ), "The number of topics in the model is different from the number of described topics"
    return model_dict


def load_document_probabilities(
    model_name: str, folder: PathLike, prefix: str = "doc_probs", suffix: str = ""
):
    """
    Loads a numpy of the document probabilities for the specified topic model, together with
    topic descriptions and the document ids covered by the document topic probability matrix

    Typical use case would be when a model has been used to infer document topic probabilities
    for a set of documents that were not used in training the model, and these probabilities
    have been stored as a .npy file to save disk space.

    NB: The .npy file should follow the format '{prefix}_{model_name}_{suffix}.npy'

    Args:
        model_name: String name of the model
        folder: Location of the model files
        prefix: Prefix of the filename of the document probability .npy file
        suffix: Suffix of the filename of the document probability .npy file

    Returns:
        Dictionary with the following keys:
            - doc_probs: A numpy array of shape (n_documents, n_topics)
            - document_ids: A list of document ids
            - topic_descriptions: A table with topic descriptions
    """
    # Figure out the file name
    _suffix = f"_{suffix}" if suffix != "" else suffix
    filename = f"{prefix}_{model_name}{_suffix}"
    # Load in relevant data
    doc_probs_dict = {
        "model_name": model_name,
        "doc_probs": np.load(folder / f"{filename}.npy"),
        "document_ids": read_text_items(folder / f"{filename}_document_ids.txt"),
        "topic_descriptions": load_topic_descriptions(model_name, folder),
    }
    # Data consistency checks
    assert doc_probs_dict["doc_probs"].shape[0] == len(
        doc_probs_dict["document_ids"]
    ), "The number of documents of the probability matrix is different from the number of document identifiers"
    assert doc_probs_dict["doc_probs"].shape[1] == len(
        doc_probs_dict["topic_descriptions"]
    ), "The number of topics of the probability matrix is different from the number of described topics"
    return doc_probs_dict


def get_document_topic_distributions(mdl: LDAModel):
    """
    Gets document topic distributions.

    Args:
        mdl: Tomotopy topic model (usually LDAModel)

    Returns:
        Numpy array with the shape (n_documents, n_topics) where each
        element (i, j) is the probability of the document i belonging
        to the topic j
    """
    return np.stack([doc.get_topic_dist() for doc in mdl.docs])


def topic_name(topic_id: int) -> str:
    """Conventional format for naming topics in tables"""
    return f"topic_{topic_id}"


def create_document_topic_probability_table(
    document_ids: Iterator[str],
    topic_model: LDAModel = None,
    document_topic_probabilities: ArrayLike = None,
) -> pd.DataFrame:
    """
    Generates a data frame with a column for document identifiers, and a column
    for each topic indicating each document's topic probabilities. Document probabilities
    can be sourced either from the topic model or a document probability matrix.

    Args:
        document_ids: A list of document ids
        topic_model: A tomotopy topic model
        document_topic_probabilities: Array of topic probabilities with the shape (n_documents, n_topics)

    Returns:
        Data frame with columns:
            - 'id': document identifiers
            - 'topic_{topic_id}': probabilities of documents belonging to topic
                specified by the topic_id number
    """
    if topic_model is not None:
        document_topic_probabilities = get_document_topic_distributions(topic_model)
    # Data consistency check
    assert (
        len(document_ids) == document_topic_probabilities.shape[0]
    ), "Number of document identifiers does not match the number of documents of the probability matrix"
    return (
        pd.DataFrame(
            data=document_topic_probabilities,
            index=document_ids,
            columns=[
                topic_name(i) for i in range(document_topic_probabilities.shape[1])
            ],
        )
        .reset_index()
        .rename(columns={"index": "id"})
    )


def query_topics(
    topics: Iterator[int], document_topic_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns topic probabilities for each document

    Args:
        topics: A list of topic numbers
        topic_model: Tomotopy topic model (usually LDAModel)

    Returns:
        A dataframe with the following columns:
            - a column for document identifiers
            - a column for each of the topic indicating documents' topic probabilities;
                the column name follows the format...
    """
    column_names = [topic_name(i) for i in topics]
    return document_topic_table[["id"] + column_names]


def make_pyLDAvis(topic_model: LDAModel, output_filepath: PathLike):
    """Creates an html file for exploring topic model results"""
    pyLDAvis.save_html(
        pyLDAvis.prepare(
            topic_term_dists=get_topic_token_distributions(topic_model),
            doc_topic_dists=get_document_topic_distributions(topic_model),
            doc_lengths=np.array([len(doc.words) for doc in topic_model.docs]),
            vocab=list(topic_model.used_vocabs),
            term_frequency=topic_model.used_vocab_freq,
            start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
            sort_topics=False,  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
        ),
        str(output_filepath),
    )


class QueryTopics:
    """
    This class helps to query documents based on their topic probabilities
    """

    def __init__(self, model_data: dict):
        """
        Args:
            model_data is a dictionary following the format of the outputs
            of either load_document_probabilities() or load_lda_model_data()
        """
        self.model_name = model_data["model_name"]
        self.topic_descriptions = model_data["topic_descriptions"]
        self.document_ids = model_data["document_ids"]
        self.topic_model = model_data["model"] if "model" in model_data else None
        self.doc_prob_table = create_document_topic_probability_table(
            document_ids=self.document_ids,
            topic_model=self.topic_model,
            document_topic_probabilities=model_data["doc_probs"]
            if "doc_probs" in model_data
            else None,
        )

    def check_topics(self, topics: Iterator[int]):
        """Return the topic probabilities and document ids of the specified topics"""
        return query_topics(topics, self.doc_prob_table)

    def sort_documents_by_topic(self, topic_id: int):
        """
        Sort documents according to their probability of belonging
        to the specified topic id
        """
        return query_topics([topic_id], self.doc_prob_table).sort_values(
            topic_name(topic_id), ascending=False
        )
