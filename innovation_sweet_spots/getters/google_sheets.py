"""
innovation_sweet_spots.getters.google_sheets

Module for easy access to Google Sheets

"""
from innovation_sweet_spots import PROJECT_DIR, logging
import pandas as pd
from pathlib import Path
from innovation_sweet_spots.utils.google_sheets import download_google_sheet


def get_sheet_data(
    local_path: Path,
    google_sheet_id: str,
    wks_name: str,
    from_local=True,
    save_locally=True,
) -> pd.DataFrame:
    """Load data from either googlesheets or csv locally

    Args:
        local_path: Path to csv
        google_sheet_id: Google Sheets id, e.g '1O1PA6TvHHVyX1-lMAMoGYUhecQToGQpqv3l9GxXMbB8'
        wks_name: Worksheet to load from Google Sheet
        from_local: Set to True to load from local_path
        save_locally: Set to True to save data loaded from
            Google Sheets to local_path.

    Returns:
        Dataframe of loaded data
    """
    if from_local:
        logging.info(f"Loading data from {local_path}")
        return pd.read_csv(local_path)
    else:
        data = download_google_sheet(google_sheet_id, wks_name)
        if save_locally:
            data.to_csv(local_path, index=False)
            logging.info(f"Sheet saved locally to {local_path}")
        return data


def get_foodtech_search_terms(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get search terms for food tech project"""
    return (
        get_sheet_data(
            local_path=PROJECT_DIR
            / "inputs/data/misc/foodtech/foodtech_search_terms.csv",
            google_sheet_id="1O1PA6TvHHVyX1-lMAMoGYUhecQToGQpqv3l9GxXMbB8",
            wks_name="search_terms",
            from_local=from_local,
            save_locally=save_locally,
        )
        .reset_index()
        .rename(columns={"index": "Category"})
    )


def get_foodtech_reviewed_vc(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get reviewed VC data for food tech project"""
    return get_sheet_data(
        local_path=PROJECT_DIR / "outputs/foodtech/interim/foodtech_reviewed_VC.csv",
        google_sheet_id="12D6cQXqMG9ou6XJbPpNw7S7tnr8nXK14r8XTD1BW5Fg",
        wks_name="selected_companies_v2022_08_05",
        from_local=from_local,
        save_locally=save_locally,
    )


def get_foodtech_reviewed_gtr(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get reviewed GtR data for food tech project"""
    return (
        get_sheet_data(
            local_path=PROJECT_DIR
            / "outputs/foodtech/interim/foodtech_reviewed_gtr.csv",
            google_sheet_id="1ZZQO6m6BSIiwTqgfHq9bNaf_FB1HG4EqwedgDLzESa0",
            wks_name="ukri",
            from_local=from_local,
            save_locally=save_locally,
        )
        .reset_index()
        .rename(columns={"index": "id"})
    )


def get_foodtech_reviewed_nihr(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get reviewed NIHR data for food tech project"""
    return (
        get_sheet_data(
            local_path=PROJECT_DIR
            / "outputs/foodtech/interim/foodtech_reviewed_nihr.csv",
            google_sheet_id="1ZZQO6m6BSIiwTqgfHq9bNaf_FB1HG4EqwedgDLzESa0",
            wks_name="nihr",
            from_local=from_local,
            save_locally=save_locally,
        )
        .reset_index()
        .rename(columns={"index": "id"})
    )


def get_foodtech_heat_map(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get heatmap data for food tech project"""
    return (
        get_sheet_data(
            local_path=PROJECT_DIR / "outputs/foodtech/interim/foodtech_heatmap.csv",
            google_sheet_id="1SX_5jBSNtegyxVFo4CGvBoV0pPykZ0EKz8hui0kpLSo",
            wks_name="heatmap",
            from_local=from_local,
            save_locally=save_locally,
        )
        .reset_index()
        .rename(columns={"index": "Category"})
    )


def get_foodtech_guardian(
    from_local: bool = True, save_locally: bool = True
) -> pd.DataFrame:
    """Get guardian data for food tech project"""
    return (
        get_sheet_data(
            local_path=PROJECT_DIR / "outputs/foodtech/interim/guardian_final_hits.csv",
            google_sheet_id="1UuYDhLtkMt7RFN2WR9AcXTOS5rUmokhD7Cqdam1Nrsk",
            wks_name="guardian_final_hits",
            from_local=from_local,
            save_locally=save_locally,
        )
        .reset_index()
        .rename(columns={"index": "id"})
    )
