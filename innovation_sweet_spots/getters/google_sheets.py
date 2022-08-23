"""
innovation_sweet_spots.getters.google_sheets

Module for easy access to Google Sheets

"""
from innovation_sweet_spots import PROJECT_DIR, logging
import pandas as pd
import pickle
import os.path
import dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from typing import Iterable
from pathlib import Path


def get_credentials_path() -> Path:
    """Finds the path to the credentials file"""
    if os.path.isfile(PROJECT_DIR / ".env"):
        # Load the .env file
        dotenv.load_dotenv(PROJECT_DIR / ".env")
        try:
            # Try fetching the path to credentials
            return Path(os.environ["GOOGLE_SHEETS_CREDENTIALS"])
        except:
            pass
    # If no .env file or if the key is not in the .env file
    logging.warning("No credentials found!")
    return None


def gsheet_api_check(scopes: Iterable[str]):
    """Authorise access to Google Sheets"""
    creds = None
    credentials_path = get_credentials_path()
    token_path = credentials_path.parent / "token.pickle"

    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)
    return creds


def pull_sheet_data(scopes: Iterable[str], spreadsheet_id: str, data_range: str):
    """Get data from Google Sheets"""
    creds = gsheet_api_check(scopes)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    result = (
        sheet.values().get(spreadsheetId=spreadsheet_id, range=data_range).execute()
    )
    values = result.get("values", [])

    if not values:
        logging.warning("No data found.")
    else:
        rows = (
            sheet.values().get(spreadsheetId=spreadsheet_id, range=data_range).execute()
        )
        data = rows.get("values")
        logging.info("COMPLETE: Data copied")
        return data


def get_foodtech_search_terms(from_local=True, save_locally=True):
    """Get search terms for food tech project"""
    local_path = PROJECT_DIR / "inputs/data/misc/foodtech/foodtech_search_terms.csv"
    # Google sheets info
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    spreadsheet_id = "1O1PA6TvHHVyX1-lMAMoGYUhecQToGQpqv3l9GxXMbB8"
    data_range = "search_terms"
    if from_local:
        logging.info(f"Loading search terms from {local_path}")
        return pd.read_csv(local_path)
    else:
        data = pull_sheet_data(scopes, spreadsheet_id, data_range)
        data_df = pd.DataFrame(data[1:], columns=data[0])
        if save_locally:
            data_df.to_csv(local_path, index=False)
            logging.info(f"Search terms saved locally to {local_path}")
        return data_df
