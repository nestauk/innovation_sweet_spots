import os.path
import dotenv
from innovation_sweet_spots import PROJECT_DIR, logging
from pathlib import Path
import oauth2client
import google
import pandas as pd
from df2gspread import utils as d2g_utils
from df2gspread import gspread2df as g2d
from df2gspread import df2gspread as d2g
import gspread


def _is_valid_credentials(credentials):
    """Function to monkey patch df2gspread.utils._is_valid_credentials."""
    return isinstance(
        credentials,
        (oauth2client.client.OAuth2Credentials, google.oauth2.credentials.Credentials),
    )


def get_credentials_path() -> Path:
    """Finds the path to the credentials file."""
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


# Monkey patch function and constant
d2g_utils._is_valid_credentials = _is_valid_credentials
d2g_utils.CLIENT_SECRET_FILE = get_credentials_path()


def download_google_sheet(google_sheet_id: str, wks_name: str) -> pd.DataFrame:
    """Download data from a Google Sheet as a dataframe.

    Args:
        google_sheet_id: Google sheet id, for example
            "1JHJ2VP9axSCg9DES_HYRxbsACKpLUKG_cHMe47f8iWI".
        wks_name: Name of the worksheet to download from,
            for example "Sheet1".

    Returns:
        Dataframe of data from specified Google Sheet.
    """
    return g2d.download(
        gfile=google_sheet_id, wks_name=wks_name, col_names=True, row_names=True
    )


def upload_to_google_sheet(
    df: pd.DataFrame,
    google_sheet_id: str,
    wks_name: str = "Sheet1",
    overwrite: bool = False,
):
    """Upload a dataframe to a Google Sheet.

    Args:
        df: Pandas dataframe.
        google_sheet_id: Already existing google sheet id,
            for example "1JHJ2VP9axSCg9DES_HYRxbsACKpLUKG_cHMe47f8iWI".
        wks_name: Name of the worksheet to upload to, defaults to "Sheet1".
            Note, if the worksheet already exists, this function will overwrite
            the existing worksheet with the specifed dataframe.
    """
    logging.warn(
        f"Uploading will overwrite any existing data on worksheet {wks_name} on Google Sheet with id {google_sheet_id}."
    )
    if overwrite is False:
        confirmation = input(
            "Enter 'upload' to continue with uploading the dataframe. Type anything else to not upload:"
        )
    if overwrite or (confirmation.lower() == "upload"):
        logging.info("Uploading...")
        d2g.upload(df=df, gfile=google_sheet_id, wks_name=wks_name, start_cell="A1")
        gc = gspread.authorize(d2g_utils.get_credentials())
        try:
            gc.open_by_key(google_sheet_id).__repr__()
        except gspread.client.APIError as e:
            logging.error(
                f"gspread.client.APIError: Could not upload dataframe to https://docs.google.com/spreadsheets/d/{google_sheet_id} as the Google Sheet does not exist."
            )
            raise e
        logging.info(
            f"Dataframe uploaded to https://docs.google.com/spreadsheets/d/{google_sheet_id}"
        )
    else:
        logging.info("Stopped uploading to Google Sheet.")
