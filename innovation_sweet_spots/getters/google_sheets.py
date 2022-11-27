"""
innovation_sweet_spots.getters.google_sheets

Module for easy access to Google Sheets

"""
from innovation_sweet_spots import PROJECT_DIR, logging
import pandas as pd
from googleapiclient.discovery import build
from pathlib import Path
from innovation_sweet_spots.utils.google_sheets import get_credentials


def get_sheet_from_googlesheets(spreadsheet_id: str, data_range: str) -> list:
    """Get data from Google Sheets"""
    creds = get_credentials()
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    result = (
        sheet.values().get(spreadsheetId=spreadsheet_id, range=data_range).execute()
    )
    values = result.get("values")
    if values:
        logging.info("Data copied from googlesheets")
        return values
    else:
        logging.warning("No data found.")


def get_sheet_data(
    local_path: Path,
    spreadsheet_id: str,
    data_range: str,
    from_local=True,
    save_locally=True,
) -> pd.DataFrame:
    """Load data from either googlesheets or csv locally

    Args:
        local_path: Path to csv
        spreadsheet_id: Googlesheets id, e.g '1O1PA6TvHHVyX1-lMAMoGYUhecQToGQpqv3l9GxXMbB8'
        data_range: Sheet to load from googlesheet
        from_local: Set to True to load from local_path
        save_locally: Set to True to save data loaded from
            googlesheets to local_path.

    Returns:
        Dataframe of loaded data
    """
    if from_local:
        logging.info(f"Loading search terms from {local_path}")
        return pd.read_csv(local_path)
    else:
        data = get_sheet_from_googlesheets(spreadsheet_id, data_range)
        data_df = pd.DataFrame(data[1:], columns=data[0])
        if save_locally:
            data_df.to_csv(local_path, index=False)
            logging.info(f"Sheet saved locally to {local_path}")
        return data_df


def get_foodtech_search_terms(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get search terms for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "inputs/data/misc/foodtech/foodtech_search_terms.csv",
        spreadsheet_id="1O1PA6TvHHVyX1-lMAMoGYUhecQToGQpqv3l9GxXMbB8",
        data_range="search_terms",
        from_local=from_local,
        save_locally=save_locally,
    )


def get_foodtech_reviewed_vc(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get reviewed VC data for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "outputs/foodtech/interim/foodtech_reviewed_VC.csv",
        spreadsheet_id="12D6cQXqMG9ou6XJbPpNw7S7tnr8nXK14r8XTD1BW5Fg",
        data_range="selected_companies_v2022_08_05",
        from_local=from_local,
        save_locally=save_locally,
    )


def get_foodtech_reviewed_gtr(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get reviewed GtR data for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "outputs/foodtech/interim/foodtech_reviewed_gtr.csv",
        spreadsheet_id="1ZZQO6m6BSIiwTqgfHq9bNaf_FB1HG4EqwedgDLzESa0",
        data_range="ukri",
        from_local=from_local,
        save_locally=save_locally,
    )


def get_foodtech_reviewed_nihr(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get reviewed NIHR data for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "outputs/foodtech/interim/foodtech_reviewed_nihr.csv",
        spreadsheet_id="1ZZQO6m6BSIiwTqgfHq9bNaf_FB1HG4EqwedgDLzESa0",
        data_range="nihr",
        from_local=from_local,
        save_locally=save_locally,
    )


def get_foodtech_heat_map(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get heatmap data for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "outputs/foodtech/interim/foodtech_heatmap.csv",
        spreadsheet_id="1SX_5jBSNtegyxVFo4CGvBoV0pPykZ0EKz8hui0kpLSo",
        data_range="heatmap",
        from_local=from_local,
        save_locally=save_locally,
    )


def get_foodtech_guardian(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get guardian data for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "outputs/foodtech/interim/guardian_final_hits.csv",
        spreadsheet_id="1UuYDhLtkMt7RFN2WR9AcXTOS5rUmokhD7Cqdam1Nrsk",
        data_range="guardian_final_hits",
        from_local=from_local,
        save_locally=save_locally,
    )
