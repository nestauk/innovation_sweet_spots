"""
innovation_sweet_spots.getters.dealroom

Module for easy access to downloaded Dealroom data

"""
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.getters.path_utils import DEALROOM_PATH
import innovation_sweet_spots.utils.embeddings_utils as eu

# Organising Dealroom data columns by themes
COLUMN_CATEGORIES = {
    "Basic descriptors": [
        "id",
        "NAME",
        "PROFILE URL",
        "WEBSITE",
        "TAGLINE",
        "LISTS",
        "COMPANY STATUS",
        "LOGO",
        "PIC NUMBER",
        "TRADE REGISTER NUMBER",
    ],
    "Labels": [
        "TAGS",
        "B2B/B2C",
        "REVENUE MODEL",
        "INDUSTRIES",
        "SUB INDUSTRIES",
        "DELIVERY METHOD",
        "TECHNOLOGIES",
        "SDGS",
        "TECH STACK DATA (BY PREDICTLEADS)",
    ],
    "Timelines": [
        "LAUNCH DATE",
        "CLOSING DATE",
        "YEAR COMPANY BECAME UNICORN",
        "YEAR COMPANY BECAME FUTURE UNICORN",
    ],
    "Location": [
        "ADDRESS",
        "HQ REGION",
        "HQ COUNTRY",
        "HQ CITY",
        "LATITUDE",
        "LONGITUDE",
        "LOCATIONS",
        "CUSTOM HQ REGIONS",
        "FOUNDING LOCATION",
    ],
    "Team": [
        "TEAM (DEALROOM)",
        "TEAM (EDITORIAL)",
        "FOUNDERS",
        "FOUNDERS STATUSES",
        "FOUNDERS GENDERS",
        "FOUNDERS IS SERIAL",
        "FOUNDERS BACKGROUNDS",
        "FOUNDERS UNIVERSITIES",
        "FOUNDERS COMPANY EXPERIENCE",
        "FOUNDERS FIRST DEGREE",
        "FOUNDERS FIRST DEGREE YEAR",
        "FOUNDERS LINKEDIN",
        "FOUNDERS FOUNDED COMPANIES TOTAL FUNDING",
        "NUMBER OF ALUMNI EUROPEAN FOUNDERS THAT RAISED > 10M",
    ],
    "Workforce": [
        "EMPLOYEES",
        "EMPLOYEES (2016,2017,2018,2019,2020,2021,2022)",
        "EMPLOYEES IN HQ country (2016,2017,2018,2019,2020,2021,2022)",
        "EMPLOYEE RANK 3/6/12 MONTHS",
    ],
    "Investors": [
        "INVESTORS",
        "EACH INVESTOR TYPES",
        "LEAD INVESTORS",
    ],
    "Funding": [
        "TOTAL FUNDING (EUR M)",
        "TOTAL FUNDING (USD M)",
        "LAST ROUND",
        "LAST FUNDING",
        "LAST FUNDING DATE",
        "FIRST FUNDING DATE",
        "SEED YEAR",
        "OWNERSHIPS",
        "GROWTH STAGE",
        "YEARLY GROWTH (SIMILARWEB)",
        "ALEXA GROWTH (ALL TIME)",
        "INCOME STREAMS",
        "CORE SIDE VALUE",
        "TOTAL ROUNDS NUMBER",
    ],
    "Funding (detailed)": [
        "EACH ROUND TYPE",
        "EACH ROUND AMOUNT",
        "EACH ROUND CURRENCY",
        "EACH ROUND DATE",
        "EACH ROUND INVESTORS",
    ],
    "Financials": [
        "LAST KPI DATE",
        "PROFIT (2016,2017,2018,2019,2020,2021,2022,2023)",
        "PROFIT MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "EBITDA (2016,2017,2018,2019,2020,2021,2022,2023)",
        "EBITDA MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "REVENUE (2016,2017,2018,2019,2020,2021,2022,2023)",
        "REVENUE GROWTH (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "R&D MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "FINANCIALS CURRENCY",
        "VALUATION",
        "VALUATION CURRENCY",
        "VALUATION (EUR)",
        "VALUATION (USD)",
        "VALUATION DATE",
        "HISTORICAL VALUATIONS - DATES",
        "HISTORICAL VALUATIONS - VALUES EUR M",
        "HISTORICAL VALUATIONS - VALUES USD M",
        "TRADING MULTIPLES EV/REVENUE (2017,2018,2019,2020,2021,2022,2023)",
        "TRADING MULTIPLES EV/EBITDA (2017,2018,2019,2020,2021,2022,2023)",
    ],
    "Social and web": [
        "FACEBOOK LIKES",
        "TWITTER FOLLOWERS",
        "TWITTER TWEETS",
        "TWITTER FAVORITES",
        "SW TRAFFIC 6 MONTHS",
        "SW TRAFFIC 12 MONTHS",
        "ANGELLIST",
        "FACEBOOK",
        "TWITTER",
        "LINKEDIN",
        "GOOGLE PLAY LINK",
        "ITUNES LINK",
        "APP DOWNLOADS LATEST (IOS)",
        "APP DOWNLOADS 6 MONTHS (IOS)",
        "APP DOWNLOADS 12 MONTHS (IOS)",
        "APP DOWNLOADS LATEST (ANDROID)",
        "APP DOWNLOADS 6 MONTHS (ANDROID)",
        "APP DOWNLOADS 12 MONTHS (ANDROID)",
        "TRAFFIC COUNTRIES",
        "TRAFFIC SOURCES",
        "SIMILARWEB RANK 3/6/12 MONTHS",
        "APP RANK 3/6/12 MONTHS",
    ],
}


def get_foodtech_companies() -> pd.DataFrame:
    """Dataset used in the food tech themed Innovation Sweet Spots"""
    return pd.read_csv(DEALROOM_PATH / "dealroom_foodtech_2022_11_24.csv")


# Preprocessed embeddings
MODEL = "all-mpnet-base-v2"
DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"


def get_label_embeddings(model=MODEL, folder=DIR, filename="foodtech_may2022_labels"):
    return eu.Vectors(
        model_name=model,
        folder=folder,
        filename=filename,
    )


def get_company_embeddings(
    model=MODEL, folder=DIR, filename="foodtech_may2022_companies"
):
    return eu.Vectors(
        model_name=model,
        folder=folder,
        filename=filename,
    )
