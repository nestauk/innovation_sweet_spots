import pytest
from innovation_sweet_spots.analysis.analysis_utils import *
import pandas as pd
from pandas._testing import assert_frame_equal


def test_impute_empty_years():
    mock_input_data = {"year": [2000, 2002], "values": [1, 1]}
    mock_df = pd.DataFrame(mock_input_data)

    df = impute_empty_years(mock_df)
    imputed_data = {"year": [2000, 2001, 2002], "values": [1, 0, 1]}
    assert_frame_equal(df, pd.DataFrame(imputed_data))

    df = impute_empty_years(mock_df, min_year=1999)
    imputed_data = {"year": [1999, 2000, 2001, 2002], "values": [0, 1, 0, 1]}
    assert_frame_equal(df, pd.DataFrame(imputed_data))

    df = impute_empty_years(mock_df, min_year=1999, max_year=2004)
    imputed_data = {"year": list(range(1999, 2005)), "values": [0, 1, 0, 1, 0, 0]}
    assert_frame_equal(df, pd.DataFrame(imputed_data))


def test_set_def_min_max_years():
    mock_input_data = {"year": [2000, 2001, 2002], "values": [1, 1, 1]}
    mock_df = pd.DataFrame(mock_input_data)
    assert set_def_min_max_years(mock_df, min_year=None, max_year=None) == (2000, 2002)
    assert set_def_min_max_years(mock_df, min_year=2001, max_year=None) == (2001, 2002)
    assert set_def_min_max_years(mock_df, min_year=2001, max_year=2001) == (2001, 2001)


def test_gtr_deduplicate_projects():
    mock_input_data = {
        "title": ["Project A", "Project A", "Project A", "Project B"],
        "description": ["aaa", "aaa", "different aaa", "bbb"],
        "amount": [1, 2, 5, 10],
        "start": [2011, 2009, 2001, 2020],
    }
    mock_df = pd.DataFrame(mock_input_data)
    df = gtr_deduplicate_projects(mock_df)
    deduplicated_data = {
        "title": ["Project A", "Project A", "Project B"],
        "description": ["different aaa", "aaa", "bbb"],
        "amount": [5, 3, 10],
        "start": [2001, 2009, 2020],
    }
    assert_frame_equal(df, pd.DataFrame(deduplicated_data))


def test_gtr_funding_per_year():
    mock_input_data = {
        "start": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
        "project_id": ["a", "b", "c", "d"],
        "amount": [1000, 2000, 10000, 5000],
    }
    mock_df = pd.DataFrame(mock_input_data)
    aggregated_df = gtr_funding_per_year(mock_df)
    aggregated_data = {
        "year": [2001, 2002, 2003, 2004],
        "no_of_projects": [1, 2, 0, 1],
        "amount_total": [1.0, 12.0, 0.0, 5.0],
        "amount_median": [1.0, 6.0, 0.0, 5.0],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))
    # Set lower bound for years
    aggregated_df = gtr_funding_per_year(mock_df, min_year=2002)
    aggregated_data = {
        "year": [2002, 2003, 2004],
        "no_of_projects": [2, 0, 1],
        "amount_total": [12.0, 0.0, 5.0],
        "amount_median": [6.0, 0.0, 5.0],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))
    # Set lower and higher bound for years
    aggregated_df = gtr_funding_per_year(mock_df, min_year=2002, max_year=2003)
    aggregated_data = {
        "year": [2002, 2003],
        "no_of_projects": [2, 0],
        "amount_total": [12.0, 0.0],
        "amount_median": [6.0, 0.0],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))


def test_cb_orgs_founded_per_year():
    mock_input_data = {
        "id": ["a", "b", "c", "d"],
        "founded_on": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
    }
    mock_df = pd.DataFrame(mock_input_data)
    aggregated_df = cb_orgs_founded_per_year(mock_df)
    aggregated_data = {
        "year": [2001, 2002, 2003, 2004],
        "no_of_orgs_founded": [1, 2, 0, 1],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))
    # Set lower bound for years
    aggregated_df = cb_orgs_founded_per_year(mock_df, min_year=2002)
    aggregated_data = {
        "year": [2002, 2003, 2004],
        "no_of_orgs_founded": [2, 0, 1],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))
    # Set lower and higher bound for years
    aggregated_df = cb_orgs_founded_per_year(mock_df, min_year=2002, max_year=2005)
    aggregated_data = {
        "year": [2002, 2003, 2004, 2005],
        "no_of_orgs_founded": [2, 0, 1, 0],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))


def test_cb_investments_per_year():
    mock_input_data = {
        "funding_round_id": ["a", "b", "c", "d"],
        "announced_on": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
        "raised_amount_usd": [10, 20, 30, 50],
        "raised_amount_gbp": [12, 24, 36, 58],
    }
    mock_df = pd.DataFrame(mock_input_data)
    aggregated_df = cb_investments_per_year(mock_df)
    aggregated_data = {
        "year": [2001, 2002, 2003, 2004],
        "no_of_rounds": [1, 2, 0, 1],
        "raised_amount_usd_total": [10, 50, 0, 50],
        "raised_amount_gbp_total": [12, 60, 0, 58],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))
    # Set lower bound for years
    aggregated_df = cb_investments_per_year(mock_df, min_year=2002)
    aggregated_data = {
        "year": [2002, 2003, 2004],
        "no_of_rounds": [2, 0, 1],
        "raised_amount_usd_total": [50, 0, 50],
        "raised_amount_gbp_total": [60, 0, 58],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))
    # Set lower and higher bound for years
    aggregated_df = cb_investments_per_year(mock_df, min_year=2002, max_year=2005)
    aggregated_data = {
        "year": [2002, 2003, 2004, 2005],
        "no_of_rounds": [2, 0, 1, 0],
        "raised_amount_usd_total": [50, 0, 50, 0],
        "raised_amount_gbp_total": [60, 0, 58, 0],
    }
    assert_frame_equal(aggregated_df, pd.DataFrame(aggregated_data))


def test_moving_average():
    mock_input_data = {
        "year": [2000, 2001, 2002],
        "values": [1, 3, 5],
    }
    mock_df = pd.DataFrame(mock_input_data)
    ma_df = moving_average(mock_df)
    output_data = {
        "year": [2000, 2001, 2002],
        "values": [1, 3, 5],
        "values_sma3": [1.0, 2.0, 3.0],
    }
    assert_frame_equal(ma_df, pd.DataFrame(output_data))

    ma_df = moving_average(mock_df, window=2)
    output_data = {
        "year": [2000, 2001, 2002],
        "values": [1, 3, 5],
        "values_sma2": [1.0, 2.0, 4.0],
    }
    assert_frame_equal(ma_df, pd.DataFrame(output_data))

    ma_df = moving_average(mock_df, window=2, replace_columns=True)
    output_data = {
        "year": [2000, 2001, 2002],
        "values": [1.0, 2.0, 4.0],
    }
    assert_frame_equal(ma_df, pd.DataFrame(output_data))
