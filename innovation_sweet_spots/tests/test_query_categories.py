from unittest import mock
from innovation_sweet_spots.analysis.query_categories import (
    query_gtr_categories,
    query_cb_categories,
)
import pandas as pd


def category_check_side_effect(*args, **kwargs):
    if args[0] == "category_1":
        return ["id_1", "id_3"]
    elif args[0] == "category_2":
        return ["id_2"]


@mock.patch(
    "innovation_sweet_spots.analysis.query_categories.initialise_gtr_id_table",
    return_value=pd.DataFrame({"id": ["id_1", "id_2", "id_3"]}),
)
@mock.patch(
    "innovation_sweet_spots.analysis.query_categories.is_gtr_project_in_category",
    return_value=[],
    side_effect=category_check_side_effect,
)
def test_query_gtr_categories(
    mock_is_gtr_project_in_category, mock_initialise_gtr_id_table
):

    # Test with a single category
    output_df = query_gtr_categories(
        categories=["category_1"], return_only_matches=False, verbose=False
    )

    expected_df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "category_1": [True, False, True],
            "any_category": [True, False, True],
        }
    )

    assert output_df.equals(expected_df)

    # Test returning only matches
    output_df = query_gtr_categories(
        categories=["category_1"], return_only_matches=True, verbose=False
    )

    expected_df = pd.DataFrame(
        {
            "id": ["id_1", "id_3"],
            "category_1": [True, True],
            "any_category": [True, True],
        }
    )

    assert output_df.equals(expected_df)

    # Test multiple categories
    output_df = query_gtr_categories(
        categories=["category_1", "category_2"],
        return_only_matches=False,
        verbose=False,
    )

    expected_df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "category_1": [True, False, True],
            "category_2": [False, True, False],
            "any_category": [True, True, True],
        }
    )

    assert output_df.equals(expected_df)


@mock.patch(
    "innovation_sweet_spots.analysis.query_categories.initialise_cb_id_table",
    return_value = pd.DataFrame({"id": ["id_1", "id_2", "id_3"]}
)
@mock.patch(
    "innovation_sweet_spots.analysis.query_categories.is_cb_organisation_in_category",
    return_value = [],
    side_effect = category_check_side_effect
)
def test_query_cb_categories(
    mock_is_cb_organisation_in_category, mock_initialise_cb_id_table
):

    # Test with a single category
    output_df = query_cb_categories(
        categories=["category_1"], return_only_matches=False, verbose=False
    )

    expected_df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "category_1": [True, False, True],
            "any_category": [True, False, True],
        }
    )

    assert output_df.equals(expected_df)

    # Test returning only matches
    output_df = query_cb_categories(
        categories=["category_1"], return_only_matches=True, verbose=False
    )

    expected_df = pd.DataFrame(
        {
            "id": ["id_1", "id_3"],
            "category_1": [True, True],
            "any_category": [True, True],
        }
    )

    assert output_df.equals(expected_df)

    # Test multiple categories
    output_df = query_cb_categories(
        categories=["category_1", "category_2"],
        return_only_matches=False,
        verbose=False,
    )

    expected_df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "category_1": [True, False, True],
            "category_2": [False, True, False],
            "any_category": [True, True, True],
        }
    )

    assert output_df.equals(expected_df)
