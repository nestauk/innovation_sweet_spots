"""

"""
from innovation_sweet_spots.getters import guardian
from innovation_sweet_spots.utils.pd import (
    pd_data_collection_utils as dcu,
    pd_data_processing_utils as dpu,
    pd_collocation_utils as cu,
    pd_pos_utils as pos,
)
from innovation_sweet_spots.utils.io import save_pickle, load_pickle, load_json
from innovation_sweet_spots.analysis.analysis_utils import impute_empty_periods
import innovation_sweet_spots.utils.io
from innovation_sweet_spots.getters.path_utils import OUTPUT_DATA_PATH
import os
import csv
import pandas as pd
import itertools
import spacy
from collections import defaultdict
from typing import Iterator, List
import altair as alt
from tqdm import tqdm
from innovation_sweet_spots import logger
from bertopic._bertopic import BERTopic

MIN_YEAR = 2000
MAX_YEAR = 2022

# Load a language model
NLP = spacy.load("en_core_web_sm")

# Variables specific to the Guardian news
TAGS_GUARDIAN = ["p", "h2"]
METADATA_GUARDIAN = ["webUrl", "webTitle", "webPublicationDate", "tags"]

DEFAULT_OUTPUTS_DIR = OUTPUT_DATA_PATH / "discourse_analysis_outputs"


def get_guardian_articles(
    search_terms: Iterator[str],
    use_cached: bool = True,
    allowed_categories: Iterator[str] = [],
    query_identifier: str = "",
    save_outputs: bool = False,
    outputs_path=DEFAULT_OUTPUTS_DIR,
):
    """
    Fetches articles from the Guardian API and filters them based on category

    Args:
        search_terms: Query terms
        use_cached: Whether to use pre-computed intermediate outputs
        allowed_categories: Only articles from these Guardian article categories will be selected
    """
    # For each search term/phrase, download corresponding articles
    articles = [
        guardian.search_content(search_term, use_cached=use_cached)
        for search_term in search_terms
    ]
    # Combine results across set of search terms
    aggregated_articles = list(itertools.chain(*articles))
    # Only keep articles from specified Guardian news categories
    filtered_articles = dcu.filter_by_category(aggregated_articles, allowed_categories)
    # Extract article metadata
    metadata = dcu.get_article_metadata(
        filtered_articles, fields_to_extract=METADATA_GUARDIAN
    )
    # Extract article text using html tags
    document_text = (
        dcu.get_article_text_df(filtered_articles, TAGS_GUARDIAN)
        .assign(
            date=lambda df: pd.to_datetime(
                df["id"].apply(lambda x: metadata[x]["webPublicationDate"])
            )
        )
        .assign(year=lambda df: df.date.dt.year)
    )
    # Remove articles without text
    document_text = document_text[-document_text.text.isnull()]
    document_text = document_text[document_text.text != ""]
    metadata = {i: metadata[i] for i in document_text["id"].to_list()}
    # Save the filtered article text and metadata
    if save_outputs:
        outputs_path = outputs_path / query_identifier
        outputs_path.mkdir(parents=True, exist_ok=True)
        filename = f"document_text_{query_identifier}.csv"
        document_text.to_csv(
            outputs_path / filename, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        logger.info(f"Saved a table in {outputs_path / filename}")
        save_pickle(metadata, outputs_path / f"metadata_dict_{query_identifier}.pkl")
    return document_text, metadata


class DiscourseAnalysis:
    """
    A class that helps querying Guardian news API
    """

    def __init__(
        self,
        search_terms: Iterator[str],
        required_terms: Iterator[Iterator[str]] = [[]],
        banned_terms: Iterator[str] = [],
        use_cached: bool = True,
        query_identifier: str = "",
        outputs_path=DEFAULT_OUTPUTS_DIR,
        verbose=False,
    ):
        """
        Args:
            search_terms: Query terms
            required_terms: Only documents with these terms will be allowed
            banned_terms: Documents with these terms will be removed
            use_cached: Whether to use pre-computed intermediate outputs
            query_identifier: Name of the query (used to create/lookup folders and filenames)
            outputs_path: Location to serve intermediate, processed outputs
            verbose: If True, it will output logging messages

        """
        self.search_terms = search_terms
        self.use_cached = use_cached
        self.required_terms = required_terms
        self.banned_terms = banned_terms
        self.q_id = query_identifier
        self.outputs_path = outputs_path / query_identifier
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        self.v = verbose
        if not self.v:
            innovation_sweet_spots.utils.io.VERBOSE = False
        # Document data (eg, articles)
        self._all_document_text = None
        self._document_text = None
        self._metadata = None
        self.load_documents()
        self.load_metadata()
        # Intermediate outputs
        self._sentence_records = None
        self._sentence_record_dict = None
        self._processed_articles_by_year = None
        self._term_sentences = None
        self._combined_term_sentences = None
        self._noun_chunks_all_years = None
        self._flat_sentences = None
        self._list_of_flat_sentences = None
        self._list_of_pos_phrases = None
        # Load precomputed intermediate outputs
        if self.use_cached and (os.path.isdir(self.outputs_path)):
            self.load_preprocessed_data()
        # Analysis outputs
        self._sentence_mentions = None
        self._term_rank = None
        self._term_temporal_rank = None
        self._phrase_patterns = None
        self._subject_phrase_triples = None
        self._object_phrase_triples = None
        self._pos_phrases = None
        self.pos_period_length = 3

    ### Collection
    @property
    def document_text(self):
        """Documents filtered by required and banned terms"""
        if self._document_text is None:
            self._document_text = self.subset_documents(
                self.search_terms, self._all_document_text
            )
            self._document_text = self.subset_documents(
                self.required_terms, self._document_text
            )
            self._document_text = self.remove_documents(
                self.banned_terms, self._document_text
            )
        return self._document_text

    @property
    def metadata(self):
        return self._metadata

    def load_documents(
        self, load_cached: bool = True, document_text: pd.DataFrame = None
    ):
        """Loads document texts"""
        if load_cached and document_text is None:
            try:
                document_path = self.outputs_path / f"document_text_{self.q_id}.csv"
                self._all_document_text = pd.read_csv(document_path).assign(
                    date=lambda df: pd.to_datetime(df.date)
                )
            except FileNotFoundError as e:
                logger.warning(
                    f"{e}. Either create {document_path} or run load_documents with document_text variable assigned."
                )
        else:
            self._all_document_text = document_text

    def load_metadata(self, load_cached: bool = True, metadata: dict = None):
        """Loads metadata"""
        if load_cached and metadata is None:
            try:
                metadata_path = self.outputs_path / f"metadata_dict_{self.q_id}.pkl"
                self._metadata = load_pickle(metadata_path)
            except FileNotFoundError as e:
                logger.warning(
                    f"{e}. Either create {metadata_path} or run load_metadata with metadata variable assigned."
                )
        else:
            self._metadata = metadata

    def load_preprocessed_data(self):
        """Loads all preprocessed data"""
        try:
            self._processed_articles_by_year = load_pickle(
                self.outputs_path / f"processed_articles_by_year_{self.q_id}.pkl"
            )
            self._sentence_records = load_pickle(
                self.outputs_path / f"sentence_records_{self.q_id}.pkl"
            )
            self._sentence_record_dict = load_pickle(
                self.outputs_path / f"sentence_record_dict_{self.q_id}.pkl"
            )
            self._noun_chunks_all_years = load_pickle(
                self.outputs_path / f"noun_chunks_{self.q_id}.pkl"
            )
            self._term_sentences = load_pickle(
                self.outputs_path / f"term_sentences_{self.q_id}.pkl"
            )
            self._combined_term_sentences = load_pickle(
                self.outputs_path / f"combined_sentences_{self.q_id}.pkl"
            )
        except:
            logger.warning(
                f"Intermediate outputs were not found in {self.outputs_path}"
            )

    def save_preprocessed_data(self):
        """Save preprocessed outputs"""
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        save_pickle(self.metadata, self.outputs_path / f"metadata_dict_{self.q_id}.pkl")
        save_pickle(
            self.processed_articles_by_year,
            self.outputs_path / f"processed_articles_by_year_{self.q_id}.pkl",
        )
        save_pickle(
            self.sentence_records,
            self.outputs_path / f"sentence_records_{self.q_id}.pkl",
        )
        save_pickle(
            self.sentence_record_dict,
            self.outputs_path / f"sentence_record_dict_{self.q_id}.pkl",
        )
        save_pickle(
            self.noun_chunks_all_years,
            self.outputs_path / f"noun_chunks_{self.q_id}.pkl",
        )
        save_pickle(
            self.term_sentences, self.outputs_path / f"term_sentences_{self.q_id}.pkl"
        )
        save_pickle(
            self.combined_term_sentences,
            self.outputs_path / f"combined_sentences_{self.q_id}.pkl",
        )

    def subset_documents(
        self, filter_term_sets: Iterator[Iterator[str]], documents: pd.DataFrame = None
    ):
        """
        Keeps articles that incldue specified filter terms.

        Note that you can apply multiple, sequential filters by providing
        a list of list of terms; for example: [['UK','England'], ['heating', 'energy']].
        In this example, the first nested list selects articles that mention specific
        geographic locations, and the second list selects articles about specifc themes.
        """
        document_text_ = (
            self._all_document_text.copy() if documents is None else documents.copy()
        )
        # Check if sets_of_filter_terms is a list of lists
        if type(filter_term_sets[0]) is not list:
            filter_term_sets = [filter_term_sets]
        # Go through all sets of filter terms
        for filter_terms in filter_term_sets:
            document_text_ = dcu.subset_articles(document_text_, filter_terms, [])
        return document_text_

    def remove_documents(
        self, banned_terms: Iterator[str], documents: pd.DataFrame = None
    ):
        """Removes any articles that have the specified banned terms"""
        document_text_ = (
            self._all_document_text.copy() if documents is None else documents.copy()
        )
        for banned_term in banned_terms:
            document_text_ = dcu.remove_articles(document_text_, banned_term)
        return document_text_

    ### Preprocessing

    def process_documents(self):
        """"""
        (
            self._processed_articles_by_year,
            self._sentence_records,
        ) = dpu.generate_sentence_corpus_by_year(
            self.document_text, NLP, "year", "text", "id"
        )

    @property
    def sentence_records(self):
        """"""
        if self._sentence_records is None:
            self.process_documents()
        return self._sentence_records

    @property
    def sentence_record_dict(self):
        """Link sentences to article IDs"""
        if self._sentence_record_dict is None:
            self._sentence_record_dict = {
                elem[0]: elem[1] for elem in self.sentence_records
            }
        return self._sentence_record_dict

    @property
    def term_sentences(self):
        """"""
        if self._term_sentences is None:
            self._term_sentences = {
                term: dpu.get_flat_sentence_mentions([term], self.sentence_records)
                for term in self.search_terms
            }
        return self._term_sentences

    @property
    def combined_term_sentences(self):
        """"""
        if self._combined_term_sentences is None:
            self._combined_term_sentences = dpu.get_flat_sentence_mentions(
                self.search_terms, self.sentence_records
            )
        return self._combined_term_sentences

    @property
    def processed_articles_by_year(self):
        """"""
        if self._processed_articles_by_year is None:
            self.process_documents()
        return self._processed_articles_by_year

    @property
    def noun_chunks_all_years(self):
        """"""
        # Use spacy functionality to identify noun phrases in all sentences in the corpus.
        # These often provide a useful starting point for analysing language used around a given technology.
        if self._noun_chunks_all_years is None:
            self._noun_chunks_all_years = {
                str(year): dpu.get_noun_chunks(
                    processed_articles, remove_det_articles=True
                )
                for year, processed_articles in self.processed_articles_by_year.items()
            }
        return self._noun_chunks_all_years

    ### Analysing
    @property
    def document_mentions(self):
        """Counts total number of documents that mentioned any of the search terms"""
        return (
            self.document_text.groupby("year")
            .agg(documents=("id", "count"))
            .reset_index()
            .assign(
                year=lambda df: pd.to_datetime(df.year.astype("int32"), format="%Y")
            )
            .pipe(impute_empty_periods, "year", "Y", MIN_YEAR, MAX_YEAR)
            .assign(year=lambda df: df.year.dt.year)
            .astype(int)
        )

    @property
    def sentence_mentions(self):
        """Count number of sentences that mentioned each individual search term"""
        if self._sentence_mentions is None:
            mentions_s = []
            for term in self.search_terms:
                term_mentions = pd.Series(
                    {y: len(s) for y, s in self.term_sentences[term].items()}
                )
                mentions_s.append(term_mentions)
            # Collect into a single dataframe and specify column names
            mentions_df = pd.concat(mentions_s, axis=1)
            mentions_df.columns = self.search_terms
            self._sentence_mentions = (
                mentions_df.reset_index().rename(columns={"index": "year"}).astype(int)
            )
        return self._sentence_mentions

    def view_collocations_terms(self, terms: Iterator[str], count_documents=True):
        """"""
        agg = "count" if count_documents else "sum"
        dfs = [self.view_collocations(term, print_sentences=False) for term in terms]

        df_counts = pd.concat(
            [
                (
                    df.groupby(["year", "id"])
                    .count()
                    .reset_index()
                    .groupby("year")
                    .agg(counts=("sentence", agg))
                    .rename(columns={"counts": terms[i]})
                )
                for i, df in enumerate(dfs)
            ],
            axis=1,
        )
        df_docs = (
            pd.concat(dfs, ignore_index=True)
            .drop_duplicates("id")
            .groupby("year")
            .agg(documents=("sentence", "count"))
        )
        return (
            pd.concat([df_counts, df_docs], axis=1)
            .fillna(0)
            .reset_index()
            .assign(
                year=lambda df: pd.to_datetime(df.year.astype("int32"), format="%Y")
            )
            .pipe(impute_empty_periods, "year", "Y", MIN_YEAR, MAX_YEAR)
            .assign(year=lambda df: df.year.dt.year)
            .astype(int)
        )

    @property
    def term_rank(self):
        """"""
        if self._term_rank is None:
            self.calculate_term_rank()
        return self._term_rank

    def calculate_term_rank(
        self,
        mentions_threshold=1,
        coocc_threshold=1,
        token_range=(1, 3),
        use_cached=True,
    ):
        """"""
        if use_cached:
            try:
                self._term_rank = pd.read_csv(
                    self.outputs_path / f"term_rank_{self.q_id}.csv"
                )
                return self._term_rank
            except:
                pass
        # Create a {year: sentences dataframe} dictionary
        sentence_collection_df = pd.DataFrame(self.sentence_records)
        sentence_collection_df.columns = ["sentence", "id", "year"]
        sentences_by_year = {y: v for y, v in sentence_collection_df.groupby("year")}
        # PMI measures
        related_terms = defaultdict(dict)
        # Rank measures
        normalised_ranks = defaultdict(dict)
        for year, sentences in sentences_by_year.items():
            for term in self.search_terms:
                key_terms, normalised_rank = cu.get_key_terms(
                    term,
                    sentences["sentence"],
                    NLP,
                    mentions_threshold=mentions_threshold,
                    coocc_threshold=coocc_threshold,
                    token_range=token_range,
                )
                # Terms and their PMI
                related_terms[year][term] = list(key_terms.items())
                # Terms and their rank
                normalised_ranks[year][term] = list(normalised_rank.items())
        # Aggregate the pmi and rank measures across search terms
        combined_pmi = cu.combine_pmi(related_terms, self.search_terms)
        combined_ranks = cu.combine_ranks(normalised_ranks, self.search_terms)
        # Organise the results
        combined_pmi_dict = defaultdict(dict)
        for year in combined_pmi:
            for term in combined_pmi[year]:
                combined_pmi_dict[year][term[0]] = term[1]
        # Dictionary mapping search terms to frequency of mentions,
        # normalised rank and pmi in a given year.
        pmi_inters_ranks = defaultdict(dict)
        for year in combined_ranks:
            for term in combined_ranks[year]:
                if term[0] in combined_pmi_dict[year]:
                    pmi_inters_ranks[year][term[0]] = (
                        term[1],
                        combined_ranks[year][term],
                        combined_pmi_dict[year][term[0]],
                    )
        # Aggregate into one long dataframe.
        self._term_rank = cu.agg_combined_pmi_rank(pmi_inters_ranks)
        return self._term_rank

    @property
    def term_temporal_rank(self):
        """"""
        if self._term_temporal_rank is None:
            self._term_temporal_rank = cu.analyse_rank_pmi_over_time(self._term_rank)
        return self._term_temporal_rank

    def make_combined_phrase_patterns(self) -> List[dict]:
        combined_phrase_patterns = {}
        for search_term in self.search_terms:
            phrase_pattern_to_add = pos.make_phrase_patterns(search_term)
            combined_phrase_patterns = {
                **combined_phrase_patterns,
                **phrase_pattern_to_add,
            }
        return combined_phrase_patterns

    def set_phrase_patterns(self, load_patterns: bool, make_patterns: bool) -> dict:
        """Set phrase patterns by loading from json file or by making patterns
        using function pd_pod_utils.make_phrase_patterns"""
        if self._phrase_patterns is None:
            if make_patterns and load_patterns:
                logger.warning(
                    "Both make_patterns and load cannot be True. Rerun with only one variable set to True"
                )
            elif make_patterns:
                self._phrase_patterns = self.make_combined_phrase_patterns()
            elif load_patterns:
                try:
                    phrase_patterns_json_path = (
                        self.outputs_path / f"phrase_patterns_{self.q_id}.json"
                    )
                    self._phrase_patterns = load_json(phrase_patterns_json_path)
                except FileNotFoundError as e:
                    logger.warning(
                        f"{e}. Create missing file or use make_patterns=True and rerun this function."
                    )
        return self._phrase_patterns

    def term_phrases(self):
        """
        All spacy identified noun chunks that contain search terms

        To view noun chunks in a given year: term_phrases['2020']
        """
        return pos.noun_chunks_w_term(self.noun_chunks_all_years, self.search_terms)

    def get_phrases_based_on_pattern(self, pattern: str) -> pd.DataFrame:
        """Get phrases based on specified pattern

        Args:
            pattern: pattern label key in self.phrase_patterns
                e.g 'heat_pumps_adj_phrase'

        Returns:
            Dataframe containing columns for:
                year, phrase, number_of_mentions, pattern
        """
        try:
            return pos.phrase_results(
                [
                    pos.aggregate_matches(
                        pos.match_patterns_across_years(
                            self.combined_term_sentences,
                            NLP,
                            self._phrase_patterns[pattern],
                            self.pos_period_length,
                        )
                    )
                ],
                pattern,
            )
        except TypeError as e:
            logger.warning(
                f"{e}. There are no phrases set yet, run function set_phrase_patterns first."
            )

    @property
    def pos_phrases(self) -> pd.DataFrame:
        """Finds POS phrases that match all phrase patterns
        from within combined_term_sentences"""
        if self._pos_phrases is None:
            try:
                self._pos_phrases = pd.concat(
                    [
                        self.get_phrases_based_on_pattern(pattern)
                        for pattern in tqdm(self._phrase_patterns)
                    ]
                ).reset_index(drop=True)
            except TypeError as e:
                logger.warning(
                    f"{e}. There are no phrases set yet, run function set_phrase_patterns first."
                )
        return self._pos_phrases

    @property
    def list_of_pos_phrases(self):
        """List of POS phrases. This format is suitable for BERTopic analysis"""
        if self._list_of_pos_phrases is None:
            self._list_of_pos_phrases = self.pos_phrases.phrase.to_list()
        return self._list_of_pos_phrases

    @property
    def subject_phrase_triples(self):
        """"""
        if self._subject_phrase_triples is None:
            self.get_subject_verb_object_triples()
        return self._subject_phrase_triples

    @property
    def object_phrase_triples(self):
        """"""
        if self._object_phrase_triples is None:
            self.get_subject_verb_object_triples()
        return self._object_phrase_triples

    def get_subject_verb_object_triples(self):
        """
        Subject verb object triples with search
        term acting both as subject and object.
        """
        self._subject_phrase_triples = defaultdict(list)
        self._object_phrase_triples = defaultdict(list)
        for term in self.search_terms:
            for year in self.combined_term_sentences:
                given_term_sentences = self.combined_term_sentences[year]["sentence"]
                subj_triples, obj_triples = pos.get_svo_triples(
                    given_term_sentences, term, NLP
                )
                subject_phrases, object_phrases = pos.get_svo_phrases(
                    subj_triples, obj_triples
                )
                self._subject_phrase_triples[year] = subject_phrases
                self._object_phrase_triples[year] = object_phrases

    def save_analysis_results(self):
        """Saves analys to outputs path"""
        self.document_mentions.to_csv(
            self.outputs_path / f"document_mentions_{self.q_id}.csv", index=False
        )
        self.sentence_mentions.to_csv(
            self.outputs_path / f"sentence_mentions_{self.q_id}.csv", index=False
        )
        self.term_rank.to_csv(
            self.outputs_path / f"term_rank_{self.q_id}.csv", index=False
        )
        self.term_temporal_rank.to_csv(
            self.outputs_path / f"term_temporal_rank_{self.q_id}.csv", index=False
        )
        self.pos_phrases.to_csv(
            self.outputs_path / f"pos_phrases_{self.q_id}.csv", index=False
        )

    ### Inspecting

    @property
    def flat_sentences(self):
        """"""
        if self._flat_sentences is None:
            self._flat_sentences = pd.concat(
                [self.combined_term_sentences[y] for y in self.combined_term_sentences]
            )
        return self._flat_sentences

    @property
    def list_of_flat_sentences(self):
        """List of flat sentencs. This format is suitable for BERTopic analysis"""
        if self._list_of_flat_sentences is None:
            self._list_of_flat_sentences = self.flat_sentences.sentence.to_list()
        return self._list_of_flat_sentences

    def view_collocations(
        self, term: str, print_sentences: bool = False, output_to_file: bool = True
    ):
        """"""
        return dpu.view_collocations(
            dpu.check_collocations(self.flat_sentences, term),
            self.metadata,
            self.sentence_record_dict,
            output_to_file=output_to_file,
            print_sentences=print_sentences,
            output_path=self.outputs_path,
        )

    def view_phrase_sentences(
        self, phrases_by_year: dict, time_period=["2019", "2020", "2021"]
    ):
        """"""
        for y in time_period:
            pos.view_phrase_sentences(
                y,
                phrases_by_year,
                self.flat_sentences,
                self.metadata,
                self.sentence_record_dict,
                output_data=False,
            )

    def fit_topic_model(self, use_phrases: bool = False) -> BERTopic:
        """Fits a BERTopic model on specified documents

        Args:
            use_phrases: If set to True, use POS phrases. If set to False use
                sentences containing search terms.

        Returns:
            topic_model: BERTopic model
            docs: list of docs (either from POS phrases or sentences
                containing search terms)
        """
        docs = self.list_of_pos_phrases if use_phrases else self.list_of_flat_sentences
        topic_model = BERTopic()
        topic_model.fit_transform(docs)
        return topic_model, docs

    ### Visualising

    def plot_mentions(self, use_documents: bool = True):
        """Plot the number of mentions for documents/sentences
        containing the search terms over time.

        Set `use_documents` to True to use document_mentions and
        to False to use sentence_mentions.
        """
        if use_documents:
            title_lbl = "documents"
            data = self.document_mentions
        else:
            title_lbl = "sentences"
            data = self.sentence_mentions
        cols = list(data.columns)
        return (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y(
                    cols[1], title=f"Number of {title_lbl} containing search terms"
                ),
                tooltip=cols,
            )
        )
