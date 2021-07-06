import spacy
import pandas as pd
from typing import Iterator

from innovation_sweet_spots.getters import gtr
import innovation_sweet_spots.analysis.analysis_utils as iss
from innovation_sweet_spots.utils.text_cleaning_utils import (
    clean_text,
    clean_chunks,
    clean_punctuation,
    split_string,
)

DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["ner"]}


def setup_spacy_model(model_parameters=DEF_LANGUAGE_MODEL):
    """
    Load and set up a spacy language model

    Args:
        model_parameters (dict): Dictionary containing language model parameters.
        The dictionary is expected to have the following structure:
            {
                "model":
                    Spacy language model name; for example "en_core_web_sm",
                "disable":
                    Pipeline components to disable for faster processing
                    (e.g. disable=["ner"] to disable named entity recognition).
                    If not required, set it to None.
            }

    Returns:
        (spacy.Language): A spacy language model
    """
    nlp = spacy.load(model_parameters["model"])
    if model_parameters["disable"] is not None:
        nlp.select_pipes(disable=model_parameters["disable"])
    return nlp


def chunk_forms(texts, nlp, n_process=1):
    """
    Generate and process noun chunks from the provided texts

    Args:
        texts (list of str): List of input texts
        nlp (spacy.Language): Spacy language model
        n_process (int): Number of processors to use

    Yields:
        (list of str): Processed chunks for each input text string
    """
    texts = (clean_punctuation(s) for s in texts)
    docs = nlp.pipe(texts, batch_size=50, n_process=n_process)
    all_chunks = ([chunk.text for chunk in doc.noun_chunks] for doc in docs)
    return ([clean_chunks(s) for s in chunks] for chunks in all_chunks)


def document_pipeline(
    df: pd.DataFrame,
    cols: Iterator[str],
    id_col: str,
    output_path="gtr/gtr_project_clean_text.csv",
):
    """Create, preprocess and save text documents from table columns"""
    logging.info(f"Creating {len(df)} documents")
    docs = iss.create_documents_from_dataframe(
        df, cols, iss.preprocess_text_clean_sentences
    )
    # Save the dataframe
    doc_df = pd.DataFrame(data={"project_id": df[id_col], "project_text": docs})
    logging.info(f"Saved {len(doc_df)} documents in {output_path}")
    doc_df.to_csv(output_path, index=False)
    return doc_df


def document_pipeline_gtr():
    """Preprocess GTR project data and create a document for each project"""
    return document_pipeline(
        gtr.get_gtr_projects(),
        cols=["title", "abstractText", "techAbstractText"],
        id_col="project_id",
        output_path=DATA_OUTPUTS / "gtr/gtr_project_clean_text.csv",
    )
