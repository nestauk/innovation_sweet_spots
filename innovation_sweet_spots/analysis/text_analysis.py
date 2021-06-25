import spacy

DEF_LANGUAGE_MODEL = {"model": "en_core_web_sm", "disable": ["ner"]}
from innovation_sweet_spots.utils.text_cleaning_utils import (
    clean_text,
    clean_chunks,
    clean_punctuation,
    split_string,
)


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
