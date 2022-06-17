"""
innovation_sweet_spots.utils.embeddings_utils
Functions for using text embeddings
"""
from innovation_sweet_spots.utils.io import save_text_items, read_text_items
from innovation_sweet_spots.utils.io import logging
import sentence_transformers
from sentence_transformers.SentenceTransformer import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cdist
from numpy.typing import ArrayLike
from typing import Iterator, Union
from pandas import DataFrame
import os


class QueryEmbeddings:
    """
    Helper class to interrogate embeddings and find most similar texts
    to a provided input text
    """

    def __init__(
        self,
        vectors: ArrayLike,
        texts: Iterator[str],
        model: Union[str, SentenceTransformer] = None,
    ):
        """
        Find most similar vectors to the given vector
        Args:
            vectors: Numpy array of shape [n_texts, n_dimensions] encoding texts
            texts: List of text strings (len(texts) must be equal to n_texts)
            model: Either model name or model instance
        """
        self.vectors = vectors
        self.texts = texts
        self.index_to_text = dict(zip(list(range(len(texts))), texts))
        self.text_to_index = dict(zip(texts, list(range(len(texts)))))
        self.model = (
            sentence_transformers.SentenceTransformer(model)
            if type(model) is str
            else model
        )

    def find_most_similar_by_id(self, vector_id: int) -> Iterator[float]:
        """Find most similar vectors to the given vector"""
        similarities = (
            1
            - cdist(self.vectors[vector_id, :].reshape(1, -1), self.vectors, "cosine")[
                0
            ]
        )
        return self.similarities_dataframe(similarities)

    def find_most_similar(self, text: str) -> Iterator[float]:
        """Find most similar vectors to the given text"""
        if text in self.texts:
            return self.find_most_similar_by_id(self.text_to_index[text])
        elif self.model is not None:
            new_vector = self.model.encode([text])
            similarities = (
                1 - cdist(new_vector[0, :].reshape(1, -1), self.vectors, "cosine")[0]
            )
            return self.similarities_dataframe(similarities)
        else:
            raise "Provide embedding model"

    def similarities_dataframe(self, similarities: Iterator[float]) -> DataFrame:
        """Prepare outputs data frame"""
        return DataFrame(
            {
                "text": self.text_to_index.keys(),
                "similarity": similarities,
            }
        ).sort_values("similarity", ascending=False)


class Vectors:
    """
    Class to deal with text embedding vectors
    """

    def __init__(
        self,
        model_name: str,
        vector_ids=None,
        vectors=None,
        filename: str = None,
        folder=None,
    ):
        """
        Args:
            model_name: Name of the model
            vector_ids: List of identifier strings (IDs) of each vector
            vectors: Numpy array of vectors, of shape [n_documents, n_dimensions]
            filename: File prefix (use if loading vectors from disk)
            folder: Location of the vectors and their IDs
        """
        self.model_name = model_name
        self._model = None
        self.filename = filename
        self.folder = folder
        files_exist = os.path.exists(
            folder / self.filepath_vectors(filename, model_name, folder)
        )

        if (vector_ids is None) and ((filename is None) or (files_exist is False)):
            # Initialise empty vectors and ids
            self.vectors = None
            self.vector_ids = []
        else:
            if (vector_ids is None) and (filename is not None) and files_exist:
                # Load from disk
                self.load_vectors_and_ids(filename, model_name, folder)
            else:
                # Take the provided vectors and vector ids
                self.vector_ids = np.array(vector_ids)
                self.vectors = vectors
            assert (
                len(self.vector_ids) == self.vectors.shape[0]
            ), "Number of vector ids does not match the number of vectors"
            assert len(np.unique(self.vector_ids)) == len(
                self.vector_ids
            ), "All vector ids must be unique"

    def load_vectors_and_ids(self, filename: str, model_name: str, folder):
        """Loads in vectors and their corresponding document ids from disk"""
        self.vector_ids = np.array(
            read_text_items(self.filepath_vector_ids(filename, model_name, folder))
        )
        self.vectors = np.load(self.filepath_vectors(filename, model_name, folder))
        logging.info(
            f"Loaded vectors from {self.filepath_vectors(filename, model_name, folder)}"
        )

    @property
    def model(self):
        """Sentence transformer model"""
        if self._model is None:
            self._model = sentence_transformers.SentenceTransformer(self.model_name)
        return self._model

    def is_id_present(self, document_id: str):
        """Checks if there is a vector for this document ID"""
        return document_id in self.vector_ids

    def id_index(self, document_id: str):
        """Returns the index of the specified ID"""
        index = np.where(self.vector_ids == document_id)[0]
        if len(index) == 1:
            return index[0]
        elif len(index) == 0:
            logging.warning(f"Document id {document_id} was not found!")
            return None

    def remove_vectors(self, document_ids: Iterator[str]):
        """Removes vectors with the specified IDs"""
        # Find the indices of specified IDs
        vector_indexes = [
            self.id_index(doc_id)
            for doc_id in document_ids
            if self.is_id_present(doc_id)
        ]
        self.vectors = np.delete(self.vectors, vector_indexes, axis=0)
        self.vector_ids = np.delete(self.vector_ids, vector_indexes, axis=0)
        logging.info(f"Removed {len(vector_indexes)} vectors")

    def generate_new_vectors(
        self,
        new_document_ids: Iterator[str],
        texts: Iterator[str],
        force_update: bool = False,
    ):
        """Generates new vectors and adds them to the existing ones"""
        new_indexes = []
        # Check which ids to update
        if self.vectors is not None:
            print(len(self.vectors))
        for i, new_id in enumerate(new_document_ids):
            if force_update or (self.is_id_present(new_id) is False):
                new_indexes.append(i)
        if (len(new_indexes) > 0) and (self.vectors is not None):
            self.add_vectors(
                new_document_ids=np.array(new_document_ids)[new_indexes],
                new_vectors=self.model.encode(np.array(texts)[new_indexes]),
            )
        elif self.vectors is None:
            # First time
            self.vectors = self.model.encode(np.array(texts)[new_indexes])
            self.vector_ids = np.array(new_document_ids)[new_indexes]
        print(len(self.vectors))
        logging.info(f"Added {len(new_indexes)} new vectors")

    def add_vectors(
        self, new_document_ids: Iterator[str], new_vectors: ArrayLike
    ) -> None:
        """Adds new document vectors"""
        assert (
            new_vectors.shape[1] == self.vectors.shape[1]
        ), "New vectors must have the same dimensions"
        new_indexes = []
        for i, new_id in enumerate(new_document_ids):
            # If ID already exists, update the existing vector
            if self.is_id_present(new_id):
                self.vectors[self.id_index(new_id), :] = new_vectors[i, :]
            # Otherwise, keep track of the new IDs
            else:
                new_indexes.append(i)
        # Concatenate the set of new IDs to the existing vectors
        self.vectors = np.concatenate(
            [self.vectors, new_vectors[new_indexes, :]], axis=0
        )
        self.vector_ids = np.concatenate(
            [self.vector_ids, np.array(new_document_ids[new_indexes])]
        )
        logging.info(
            f"Added {len(new_indexes)} new vectors and updated {len(new_document_ids)-len(new_indexes)} existing vectors"
        )

    def get_missing_ids(self, document_ids: Iterator[str]) -> Iterator[str]:
        """Returns IDs that don't have corresponding vectors"""
        return [
            doc_id for doc_id in document_ids if (self.is_id_present(doc_id) is False)
        ]

    def select_vectors(self, document_ids: Iterator[str]) -> ArrayLike:
        """Selects vectors by the specified ids"""
        indexes = [
            self.id_index(doc_id)
            for doc_id in document_ids
            if self.is_id_present(doc_id)
        ]
        if len(indexes) != len(document_ids):
            logging.warning("Not all IDs were found!")
        else:
            return self.vectors[indexes, :]

    @staticmethod
    def filepath_vectors(filename, model_name, folder) -> str:
        """Default filepath to the vector binary file"""
        return folder / f"{filename}_{model_name}.npy"

    @staticmethod
    def filepath_vector_ids(filename, model_name, folder) -> str:
        """Default filepath to the vector id file"""
        return folder / f"{filename}_{model_name}_ids.txt"

    def save_vectors(self, filename=None, folder=None) -> None:
        """Saves vectors and the corresponding document ids locally"""
        filename = self.filename if filename is None else filename
        folder = self.folder if folder is None else folder
        # Save vectors
        vectors_filepath = self.filepath_vectors(filename, self.model_name, folder)
        np.save(vectors_filepath, self.vectors)
        logging.info(f"Saved {len(self.vectors)} vectors in {vectors_filepath}")
        # Save IDs
        save_text_items(
            list(self.vector_ids),
            self.filepath_vector_ids(filename, self.model_name, folder),
        )
