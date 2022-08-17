# %%
"""
Metaflow to tokenise Gateway to Research project descriptions
NB: Takes about XX hrs for 120k+ projects

Usage (running from the terminal):
python innovation_sweet_spots/pipeline/preprocessing/flow_tokenise_nihr.py run --test-mode False
"""
from metaflow import FlowSpec, Parameter, step
from innovation_sweet_spots import PROJECT_DIR, logging
import pandas as pd

# %%
# Data parameters
NIHR_PATH = PROJECT_DIR / "inputs/data/nihr/nihr_summary_data.csv"
COLUMNS = ["project_title", "scientific_abstract", "plain_english_abstract"]

# %%
# Phraser parameters
NGRAMS = 4
MIN_COUNT = 5
THRESH = 0.35

# %%
# Outputs parameters
OUTPUT_DIR = PROJECT_DIR / "outputs/preprocessed"
OUTPUT_NAME = "nihr_abstracts_v2022_08_17"


# %%
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
        """Loads NIHR data and creates text documents from specified columns"""
        from innovation_sweet_spots.utils.text_processing_utils import (
            create_documents_from_dataframe,
        )

        # Load in Gateway to Research
        logging.info("Loading in NIHR data")
        data = pd.read_csv(NIHR_PATH)
        # Subsample if in test mode
        data = data.iloc[0:100] if self.test_mode else data

        # Combine specified columns into text documents
        self.text_documents = create_documents_from_dataframe(data, columns=COLUMNS)
        self.ids = data.recordid.to_list()

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
