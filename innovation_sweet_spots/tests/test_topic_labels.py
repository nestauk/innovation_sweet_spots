from unittest import mock, TestCase
from tempfile import NamedTemporaryFile
import json
from pathlib import Path
from innovation_sweet_spots.pipeline.give_topics_names import (
    get_wiki_topic_labels,
    get_labels,
)


@mock.patch("innovation_sweet_spots.pipeline.give_topics_names.suggest_labels")
def test_get_wiki_topic_labels(mock_suggest_labels):
    mock_suggest_labels.return_value = ["Mock Topic", "Another Mock Topic"]
    mock_terms = [["term_1", "term_2"], ["term_3", "term_4"]]
    labels = get_wiki_topic_labels(mock_terms)
    assert type(labels) is list
    assert len(labels) == len(mock_terms)
    assert labels == [mock_suggest_labels.return_value] * len(mock_terms)


@mock.patch("innovation_sweet_spots.pipeline.give_topics_names.get_wiki_topic_labels")
def test_get_labels(mock_get_wiki_topic_labels):
    mock_cluster_terms = [
        {"id": 1, "terms": ["term_1", "term_2"]},
        {"id": 2, "terms": ["term_3", "term_4"]},
    ]
    mock_labels = ["Mock Topic", "Another Mock Topic"]
    mock_get_wiki_topic_labels.return_value = [mock_labels] * len(mock_cluster_terms)

    mock_output = mock_cluster_terms.copy()
    for i in range(len(mock_output)):
        mock_output[i]["labels"] = mock_labels

    with NamedTemporaryFile() as test_file:
        json.dump(mock_cluster_terms, open(Path(test_file.name), "w"))

        output = get_labels(
            Path(test_file.name), outfile=Path(test_file.name), save=False
        )
        assert type(output) is list
        assert type(output[0]) is dict
        assert "labels" in output[0].keys()
        assert output == mock_output

        loaded_output = json.load(open(f"{Path(test_file.name)}", "r"))
        assert loaded_output == mock_output
        assert "labels" in loaded_output[0].keys()
