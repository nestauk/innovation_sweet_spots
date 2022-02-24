from innovation_sweet_spots.analysis.wrangling_utils import *
import pandas as pd
from datetime import date
import pytest

## Testing GtrWrangler ##

# Mock GtR project data
MOCK_PROJECT_IDS = ["proj_1", "proj_2"]
MOCK_PROJECT_ABSTRACTS = ["abstract", "different abstract"]
MOCK_PROJECT_START = ["2020-01-01 01:00:00", "2022-01-01 01:00:00"]
MOCK_FUND_IDS = ["fund_1", "fund_2", "fund_3", "fund_4"]
# Mock GtR project dataframe
MOCK_GTR_PROJECTS = pd.DataFrame(
    {
        "project_id": MOCK_PROJECT_IDS,
        "abstractText": MOCK_PROJECT_ABSTRACTS,
        "start": MOCK_PROJECT_START,
    }
)
# Mock links table between GtR projects and funds
MOCK_LINK_GTR_FUNDS = pd.DataFrame(
    {
        "project_id": MOCK_PROJECT_IDS + MOCK_PROJECT_IDS,
        "id": MOCK_FUND_IDS,
        "rel": ["FUND", "FUND", "FUND", "FUND"],
        "table_name": ["gtr_funds", "gtr_funds", "gtr_funds", "gtr_funds"],
    }
)
# Mock GtR fund table
MOCK_GTR_FUNDS = pd.DataFrame(
    {
        "id": MOCK_FUND_IDS,
        "end": [
            "2021-07-01 01:00:00",
            "2022-07-01 01:00:00",
            "2023-07-01 01:00:00",
            "2024-07-01 01:00:00",
        ],
        "start": [
            "2020-01-01 01:00:00",
            "2021-01-01 01:00:00",
            "2022-01-01 01:00:00",
            "2023-01-01 01:00:00",
        ],
        "category": [
            "INCOME_ACTUAL",
            "INCOME_ACTUAL",
            "INCOME_ACTUAL",
            "INCOME_ACTUAL",
        ],
        "amount": ["0", "1000", "2000", "3000"],
        "currencyCode": ["GBP", "GBP", "GBP", "GBP"],
    }
)
# Mock alternative GtR fund table (retrieved directly via API)
MOCK_GTR_FUNDS_API = pd.DataFrame(
    {
        "project_id": MOCK_PROJECT_IDS,
        "amount": [1000, 2000],
        "currencyCode": ["GBP", "GBP"],
    }
)


def create_mock_expected_split_data(
    frequency: str,
    amount_1: float,
    amount_2: float,
    start_1: str = "2020/1/1",
    start_2: str = "2021/1/1",
    end_1: str = "2023/7/1",
    end_2: str = "2024/7/1",
) -> pd.DataFrame:
    """Generate mock dataframe of expected split data

    Args:
        frequency: Time periods frequency ('Y', 'Q' or 'M')
        amount_1: Split amount for the first project
        amount_2: Split amount for the second project
        start_1: Start date for the first project. Defaults to "2020/1/1".
        start_2: Start date for the second project. Defaults to "2021/1/1".
        end_1: End date for the first project. Defaults to "2023/7/1".
        end_2: End date for the second project. Defaults to "2024/7/1".

    Returns:
        Mock dataframe of expected split data
    """
    exp_periods_1 = pd.period_range(
        start=start_1, end=end_1, freq=frequency
    ).to_timestamp()
    exp_periods_2 = pd.period_range(
        start=start_2, end=end_2, freq=frequency
    ).to_timestamp()
    exp_periods_combined = exp_periods_1.append(exp_periods_2)
    exp_n_periods_1 = len(exp_periods_1)
    exp_n_periods_2 = len(exp_periods_2)
    return pd.DataFrame(
        {
            "project_id": [MOCK_PROJECT_IDS[0]] * exp_n_periods_1
            + [MOCK_PROJECT_IDS[1]] * exp_n_periods_2,
            "abstractText": [MOCK_PROJECT_ABSTRACTS[0]] * exp_n_periods_1
            + [MOCK_PROJECT_ABSTRACTS[1]] * exp_n_periods_2,
            "start": exp_periods_combined,
            "amount": [amount_1] * exp_n_periods_1 + [amount_2] * exp_n_periods_2,
            "currencyCode": ["GBP"] * len(exp_periods_combined),
        }
    ).astype({"amount": float})


def prepare_GtrWrangler():
    """Helper function to prepare a GtrWrangler instance with mock data"""
    wrangler = GtrWrangler()
    wrangler._link_gtr_funds = MOCK_LINK_GTR_FUNDS
    wrangler._gtr_funds = MOCK_GTR_FUNDS
    wrangler._link_gtr_funds_api = MOCK_GTR_FUNDS_API
    return wrangler


def test_get_project_funds_api():
    wrangler = prepare_GtrWrangler()
    output_df = wrangler.get_project_funds_api(MOCK_GTR_PROJECTS)
    expected_df = pd.DataFrame(
        {
            "project_id": MOCK_PROJECT_IDS,
            "abstractText": MOCK_PROJECT_ABSTRACTS,
            "start": MOCK_PROJECT_START,
            "amount": [1000, 2000],
            "currencyCode": ["GBP", "GBP"],
        }
    )
    assert output_df.equals(expected_df)


def test_get_start_end_dates():
    wrangler = prepare_GtrWrangler()
    output_df = wrangler.get_start_end_dates(MOCK_GTR_PROJECTS)
    expected_df = pd.DataFrame(
        {
            "project_id": MOCK_PROJECT_IDS,
            "abstractText": MOCK_PROJECT_ABSTRACTS,
            "start": MOCK_PROJECT_START,
            "fund_start": ["2020-01-01 01:00:00", "2021-01-01 01:00:00"],
            "fund_end": ["2023-07-01 01:00:00", "2024-07-01 01:00:00"],
        }
    ).astype({"fund_start": "datetime64[ns]", "fund_end": "datetime64[ns]"})
    assert output_df.equals(expected_df)


def test_get_project_funds():
    wrangler = prepare_GtrWrangler()
    output_df = wrangler.get_project_funds(MOCK_GTR_PROJECTS)
    expected_df = (
        pd.DataFrame(
            {
                "project_id": MOCK_PROJECT_IDS + MOCK_PROJECT_IDS,
                "abstractText": MOCK_PROJECT_ABSTRACTS + MOCK_PROJECT_ABSTRACTS,
                "start": MOCK_PROJECT_START + MOCK_PROJECT_START,
                "id": MOCK_FUND_IDS,
                "fund_end": [
                    "2021-07-01 01:00:00",
                    "2022-07-01 01:00:00",
                    "2023-07-01 01:00:00",
                    "2024-07-01 01:00:00",
                ],
                "fund_start": [
                    "2020-01-01 01:00:00",
                    "2021-01-01 01:00:00",
                    "2022-01-01 01:00:00",
                    "2023-01-01 01:00:00",
                ],
                "category": [
                    "INCOME_ACTUAL",
                    "INCOME_ACTUAL",
                    "INCOME_ACTUAL",
                    "INCOME_ACTUAL",
                ],
                "amount": ["0", "1000", "2000", "3000"],
                "currencyCode": ["GBP", "GBP", "GBP", "GBP"],
            }
        )
        # Datetime types
        .astype({"fund_start": "datetime64[ns]", "fund_end": "datetime64[ns]"})
        # Note the sort order by projects
        .sort_values(["project_id", "id"]).reset_index(drop=True)
    )
    assert output_df.equals(expected_df)


### Testing CrunchbaseWrangler ###


def test_convert_deal_currency_to_gbp():
    mock_input_df = pd.DataFrame(
        {
            "announced_on_date": [date(2001, 1, 21)],
            "raised_amount": [1000],
            "raised_amount_currency_code": ["USD"],
            "raised_amount_usd": [1000],
        }
    )
    expected_df = mock_input_df.copy()
    expected_df["raised_amount_gbp"] = [684.8]
    output_df = CrunchbaseWrangler.convert_deal_currency_to_gbp(mock_input_df)
    output_df["raised_amount_gbp"] = output_df["raised_amount_gbp"].round(1)
    assert output_df.equals(expected_df)


def test_split_funding_data_output():
    wrangler = prepare_GtrWrangler()
    gtr_projects = wrangler.get_funding_data(MOCK_GTR_PROJECTS)
    expected_split_by_year = create_mock_expected_split_data("Y", 250, 500)
    expected_split_by_month = create_mock_expected_split_data(
        "M", 23.2558139534883, 46.5116279069767
    )
    expected_split_by_quarter = create_mock_expected_split_data(
        "Q", 66.6666666666666, 133.333333333333
    )
    pd.testing.assert_frame_equal(
        wrangler.split_funding_data(gtr_projects, "year"), expected_split_by_year
    )
    pd.testing.assert_frame_equal(
        wrangler.split_funding_data(gtr_projects, "month"), expected_split_by_month
    )
    pd.testing.assert_frame_equal(
        wrangler.split_funding_data(gtr_projects, "quarter"), expected_split_by_quarter
    )


def test_split_funding_data_invalid_time_period():
    wrangler = prepare_GtrWrangler()
    gtr_projects = wrangler.get_funding_data(MOCK_GTR_PROJECTS)
    with pytest.raises(ValueError):
        wrangler.split_funding_data(gtr_projects=gtr_projects, time_period="day")


### Testing other functions


def test_split_comma_seperated_string():
    assert split_comma_seperated_string("A, B,c") == ["A", "B", "c"]
    assert split_comma_seperated_string("a,,b") == ["a", "", "b"]
    assert split_comma_seperated_string(111) == []


def test_is_string_in_list():
    list_of_strings = ["a", "b"]
    assert is_string_in_list(list_of_strings, list_to_check=["a", "c"]) is True
    assert is_string_in_list(list_of_strings, list_to_check=["a", "b"]) is True
    assert is_string_in_list(list_of_strings, list_to_check=["b", "c"]) is True
    assert is_string_in_list(list_of_strings, list_to_check=["c", "d"]) is False
