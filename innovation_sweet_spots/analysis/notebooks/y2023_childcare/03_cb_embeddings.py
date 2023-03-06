# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# %%
"""
A metaflow flow to generate the embeddings for the company business descriptions
"""
import metaflow as mf
from metaflow import FlowSpec, Parameter, step
import sentence_transformers

from typing import List
import numpy.typing as npt

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


class EmbeddingFlow(FlowSpec):
    """
    Generate embeddings from company business descriptions and industry tags
    """

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=True,
    )

    embeddings: npt.ArrayLike
    model_name: str
    org_descriptions: "DataFrame"
    org_ids: List

    @step
    def start(self):
        """
        Start the flow and load the data
        """
        from industrial_taxonomy.getters.glass import get_organisation_description
        from industrial_taxonomy.getters.glass_house import glass_companies_house_lookup

        org_descriptions = get_organisation_description()
        self.org_ids = org_descriptions.index.intersection(
            list(glass_companies_house_lookup().keys())
        ).to_list()

        nrows = 50_000 if self.test_mode and not current.is_production else None
        self.org_ids = self.org_ids[:nrows]

        self.org_descriptions = org_descriptions.loc[self.org_ids][
            "description"
        ].to_list()

        self.next(self.embed_descriptions)

    @step
    def embed_descriptions(self):
        """Apply transformer to Crunchbase descriptions"""
        from sentence_transformers import SentenceTransformer

        # from torch import cuda

        # if not cuda.is_available():
        #     raise EnvironmentError("CUDA is not available")

        self.model_name = MODEL_NAME
        encoder = SentenceTransformer(self.model_name)
        self.embeddings = encoder.encode(self.org_descriptions)

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow
        """
        pass


@project(name="industrial_taxonomy")
class GlassEmbed(FlowSpec):
    """Transform descriptions of fuzzy matched companies into embeddings.

    This uses the multi-qa-mpnet-base-dot-v1 transformer model which encodes up
    to 512 tokens per document and produces embeddings with 768 dimensions.
    It is recommended by SBERT due to its performance when benchmarked against
    other transformers for semantic search:
    https://www.sbert.net/docs/pretrained_models.html

    The model produces normalised embeddings of length 1, meaning that the dot
    and cosine products are equivalent.

    Attributes:
        embeddings: Embeddings of Glass org descriptions
        model_name: Name of pre-trained transformer model used to generate encodings
        org_descriptions: Descriptions of Glass organisations that are embedded
        org_ids: Glass IDs for orgs with embeddings (follows order of embeddings)
    """

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    embeddings: npt.ArrayLike
    model_name: str
    org_descriptions: "DataFrame"
    org_ids: List

    @step
    def start(self):
        """Load matched Glass and Companies House IDs and split into chunks
        for embedding.
        """

        from industrial_taxonomy.getters.glass import get_organisation_description
        from industrial_taxonomy.getters.glass_house import glass_companies_house_lookup

        org_descriptions = get_organisation_description()
        self.org_ids = org_descriptions.index.intersection(
            list(glass_companies_house_lookup().keys())
        ).to_list()

        nrows = 50_000 if self.test_mode and not current.is_production else None
        self.org_ids = self.org_ids[:nrows]

        self.org_descriptions = org_descriptions.loc[self.org_ids][
            "description"
        ].to_list()

        self.next(self.embed_descriptions)

    @batch(
        queue="job-queue-GPU-nesta-metaflow",
        image="metaflow-pytorch",
        # Queue gives p3.2xlarge, with:
        gpu=1,
        memory=60000,
        cpu=8,
    )
    @pip(libraries={"sentence-transformers": "2.1.0"})
    @step
    def embed_descriptions(self):
        """Apply transformer to Glass descriptions"""
        from sentence_transformers import SentenceTransformer
        from torch import cuda

        if not cuda.is_available():
            raise EnvironmentError("CUDA is not available")

        self.model_name = MODEL_NAME
        encoder = SentenceTransformer(self.model_name)
        self.embeddings = encoder.encode(self.org_descriptions)

        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    GlassEmbed()


from metaflow import FlowSpec, Parameter, step


class Tokeniser(FlowSpec):
    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=True,
    )

    output_name = Parameter(
        "outputs-name", help="Name of the outputs", type=str, default=OUTPUT_NAME
    )

    output_location = Parameter(
        "outputs-location",
        help="Location of the outputs",
        type=str,
        default=str(OUTPUT_DIR),
    )

    @step
    def start(self):
        """Loads Hansard data and creates text documents from specified columns"""
        from innovation_sweet_spots.utils.text_processing_utils import (
            create_documents_from_dataframe,
        )

        # Load in Gateway to Research
        logging.info("Loading in Hansard data")
        nrows = 200 if self.test_mode else None
        data = hansard.get_debates(nrows=nrows)

        # Combine specified columns into text documents
        self.text_documents = create_documents_from_dataframe(data, columns=COLUMNS)
        self.ids = data["id"].to_list()
        assert len(data["id"].unique()) == len(data)

        self.next(self.tokenise_texts)

    @step
    def tokenise_texts(self):
        """Tokenises texts into phrases"""
        from innovation_sweet_spots.utils.text_processing_utils import (
            process_and_tokenise_corpus,
        )

        # Tokenising texts using ngram phraser
        logging.info(
            f"Tokenising {len(self.text_documents)} documents using ngram phraser"
        )
        corpus, ngram_phraser = process_and_tokenise_corpus(
            self.text_documents, n_gram=NGRAMS, min_count=MIN_COUNT, threshold=THRESH
        )

        # Create a corpus file
        self.corpus = dict(zip(self.ids, corpus))
        self.tokeniser = ngram_phraser

        self.next(self.save_outputs)

    @step
    def save_outputs(self):
        """Saves the corpus files and tokeniser to disk"""
        from innovation_sweet_spots.utils.io import save_pickle

        save_pickle(
            self.tokeniser,
            f"{self.output_location}/{'tokeniser_{}.p'.format(self.output_name)}",
        )
        save_pickle(
            self.corpus,
            f"{self.output_location}/{'tokens_{}.p'.format(self.output_name)}",
        )

        self.next(self.end)

    @step
    def end(self):
        pass


# %%
if __name__ == "__main__":
    Tokeniser()
