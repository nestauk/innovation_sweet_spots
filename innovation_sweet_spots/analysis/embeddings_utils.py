from innovation_sweet_spots import logging, PROJECT_DIR
from sentence_transformers import SentenceTransformer
from time import time
from typing import Iterator
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
import os

# Default bert model
BERT_MODEL = "paraphrase-distilroberta-base-v1"
# Default path to output models
MODEL_PATH = PROJECT_DIR / "outputs/models"


def find_most_similar(vect_id: int, vects: np.ndarray) -> Iterator[int]:
    sims = cdist(vects[vect_id, :].reshape(1, -1), vects, "cosine")
    return list(np.argsort(sims[0]))


def find_most_similar_vect(
    vect: np.ndarray, vects: np.ndarray, metric="cosine"
) -> Iterator[int]:
    sims = cdist(vect.reshape(1, -1), vects, metric=metric)
    return list(np.argsort(sims[0]))


### BERT


def setup_bert_transformer(BERT_MODEL):
    return SentenceTransformer(bert_model)


def calculate_embeddings(
    list_of_sentences: Iterator[str], bert_transformer: SentenceTransformer, fpath=None
) -> np.ndarray:
    """Calculate sentence embeddings"""
    t = time()
    logging.info(f"Calculating {len(list_of_sentences)} embeddings")
    sentence_embeddings = np.array(bert_transformer.encode(list_of_sentences))
    logging.info(f"Done in {time()-t:.2f} seconds")
    if fpath is not None:
        np.save(sentence_embeddings, fpath)
    return sentence_embeddings


### Word2Vec


def token_2_vec(
    lists_of_tokens, use_cached=True, model_name=None, model_dir=MODEL_PATH, seed=123
):
    """Generate or load a cached word2vec model, with some hardcoded parameters"""
    model_path = f"{model_dir / model_name}.model"
    if os.path.exists(model_path) and use_cached:
        model = Word2Vec.load(model_path)
        logging.info(f"Loaded in word2vec model from {model_path}")
    else:
        # Determine the window size
        n_tokens_per_list = [len(x) for x in lists_of_tokens]
        max_window = max(n_tokens_per_list)
        # Build the model
        logging.info(
            f"Building a word2vec model from {len(lists_of_tokens)} lists of tokens."
        )
        model = Word2Vec(
            sentences=lists_of_tokens,
            size=200,
            window=max_window,
            min_count=1,
            workers=4,
            sg=1,
            seed=seed,
            iter=30,
        )
        model.save(model_path)
        logging.info(f"Saved word2vec model in {model_path}")
    return model


def get_token_vectors(model, unique_tokens):
    """Get vectors from a word2vec model"""
    token2vec_emb = [model.wv[token] for token in unique_tokens]
    token2vec_emb = np.array(token2vec_emb)
    return token2vec_emb
