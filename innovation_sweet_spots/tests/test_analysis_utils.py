import pytest
from innovation_sweet_spots.analysis.analysis_utils import *
import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal


def test_impute_empty_periods():
    mock_df = pd.DataFrame(
        {
            "time_period": ["2000-01-01", "2002-01-01"],
            "values": [1.0, 1.0],
        }
    ).astype({"time_period": "datetime64[ns]"})

    df = impute_empty_periods(mock_df, "time_period", "Y", 2000, 2002)
    imputed_data = pd.DataFrame(
        {
            "time_period": ["2000-01-01", "2001-01-01", "2002-01-01"],
            "values": [1.0, 0.0, 1.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert df.equals(imputed_data)

    df = impute_empty_periods(mock_df, "time_period", "Y", 1999, 2002)
    imputed_data = pd.DataFrame(
        {
            "time_period": ["1999-01-01", "2000-01-01", "2001-01-01", "2002-01-01"],
            "values": [0.0, 1.0, 0.0, 1.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert df.equals(imputed_data)

    df = impute_empty_periods(mock_df, "time_period", "Y", 1999, 2005)
    imputed_data = pd.DataFrame(
        {
            "time_period": [
                "1999-01-01",
                "2000-01-01",
                "2001-01-01",
                "2002-01-01",
                "2003-01-01",
                "2004-01-01",
                "2005-01-01",
            ],
            "values": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert_frame_equal(df, imputed_data)


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


def test_gtr_funding_per_period():
    mock_input_data = {
        "start": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
        "project_id": ["a", "b", "c", "d"],
        "amount": [1000, 2000, 10000, 5000],
    }
    mock_df = pd.DataFrame(mock_input_data)
    aggregated_df = gtr_funding_per_period(
        mock_df, period="Y", min_year=2001, max_year=2004
    )
    aggregated_data = pd.DataFrame(
        {
            "time_period": ["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"],
            "no_of_projects": [1.0, 2.0, 0.0, 1.0],
            "amount_total": [1.0, 12.0, 0.0, 5.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert_frame_equal(aggregated_df, aggregated_data)


def test_gtr_get_all_timeseries_period():
    mock_df = pd.DataFrame(
        {
            "project_id": ["1", "2", "3", "4"],
            "title": ["Project A", "Project A", "Project B", "Project C"],
            "description": ["aaa", "aaa", "bbb", "ccc"],
            "start": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
            "amount": [1000, 2000, 10000, 5000],
        }
    )
    output = gtr_get_all_timeseries_period(
        mock_df, period="year", min_year=2001, max_year=2004
    )
    output_data = pd.DataFrame(
        {
            "time_period": ["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"],
            "no_of_projects": [1.0, 1.0, 0.0, 1.0],
            "amount_total": [1.0, 12.0, 0.0, 5.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert_frame_equal(output, output_data)


def test_cb_orgs_founded_per_period():
    mock_input_data = {
        "id": ["a", "b", "c", "d"],
        "founded_on": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
    }
    mock_df = pd.DataFrame(mock_input_data)
    aggregated_df = cb_orgs_founded_per_period(
        mock_df, period="Y", min_year=2001, max_year=2004
    )
    aggregated_data = pd.DataFrame(
        {
            "time_period": ["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"],
            "no_of_orgs_founded": [1.0, 2.0, 0.0, 1.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert_frame_equal(aggregated_df, aggregated_data)


def test_cb_investments_per_period():
    mock_df = pd.DataFrame(
        {
            "funding_round_id": ["a", "b", "c", "d"],
            "announced_on": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
            "raised_amount_usd": [10, 20, 30, 50],
            "raised_amount_gbp": [12, 24, 36, 58],
        }
    )
    aggregated_df = cb_investments_per_period(mock_df, "Y", 2001, 2004)
    aggregated_data = pd.DataFrame(
        {
            "time_period": ["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"],
            "no_of_rounds": [1.0, 2.0, 0.0, 1.0],
            "raised_amount_usd_total": [10.0, 50.0, 0.0, 50.0],
            "raised_amount_gbp_total": [12.0, 60.0, 0.0, 58.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert_frame_equal(aggregated_df, aggregated_data)


def test_cb_get_all_timeseries():
    mock_organisation_data = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "founded_on": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
        }
    )
    mock_deal_data = pd.DataFrame(
        {
            "funding_round_id": ["a", "b", "c", "d"],
            "announced_on": ["2001-02-01", "2002-10-01", "2002-12-05", "2004-04-01"],
            "raised_amount_usd": [10, 20, 30, 50],
            "raised_amount_gbp": [12, 24, 36, 58],
        }
    )
    output = cb_get_all_timeseries(
        mock_organisation_data, mock_deal_data, "year", 2001, 2004
    )
    output_data = pd.DataFrame(
        {
            "time_period": ["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"],
            "no_of_rounds": [1.0, 2.0, 0.0, 1.0],
            "raised_amount_usd_total": [10.0, 50.0, 0.0, 50.0],
            "raised_amount_gbp_total": [12.0, 60.0, 0.0, 58.0],
            "no_of_orgs_founded": [1.0, 2.0, 0.0, 1.0],
        }
    ).astype({"time_period": "datetime64[ns]"})
    assert_frame_equal(output, output_data)


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


def test_magnitude():
    mock_input_data = {
        "year": [2000, 2001, 2002, 2003],
        "values": [1, 3, 5, 4],
    }
    mock_df = pd.DataFrame(mock_input_data)
    output = magnitude(mock_df, year_start=2001, year_end=2002)
    output_ref = pd.Series([4.0], ["values"])
    assert_series_equal(output, output_ref)
    # Different year range
    output = magnitude(mock_df, year_start=2000, year_end=2002)
    output_ref = pd.Series([3.0], ["values"])
    assert_series_equal(output, output_ref)


def test_percentage_change():
    assert percentage_change(initial_value=10, new_value=15) == 50
    assert percentage_change(20, 15) == -25
    assert percentage_change(20, 100) == 400
    assert percentage_change(20, 0) == -100


def test_smoothed_growth():
    mock_input_data = {
        "year": [2000, 2001, 2002, 2003],
        "values": [1, 3, 5, 4],
    }
    mock_df = pd.DataFrame(mock_input_data)
    output = smoothed_growth(mock_df, year_start=2001, year_end=2003, window=2)
    output_ref = pd.Series([125.0], ["values"])
    assert_series_equal(output, output_ref)
    # Different year range
    output = smoothed_growth(mock_df, year_start=2000, year_end=2002, window=3)
    output_ref = pd.Series([200.0], ["values"])
    assert_series_equal(output, output_ref)


def test_estimate_magnitude_growth():
    mock_input_data = {
        "year": [2000, 2001, 2002, 2003],
        "values": [1, 3, 5, 4],
    }
    mock_df = pd.DataFrame(mock_input_data)
    output = estimate_magnitude_growth(
        mock_df, year_start=2001, year_end=2003, window=2
    )
    output_ref = pd.DataFrame(
        {"trend": ["magnitude", "growth"], "values": [4.0, 125.0]}
    )
    assert_frame_equal(output, output_ref)
