import pandas as pd
import numpy as np
from toolz import pipe
from currency_converter import CurrencyConverter
from datetime import datetime
from typing import Iterator
import itertools
import innovation_sweet_spots.getters.gtr as gtr
import innovation_sweet_spots.getters.crunchbase as cb
from functools import lru_cache


class GtrWrangler:
    """
    This class helps linking GtR projects to other GtR data such as the
    project funding or the participating organisations
    """

    def __init__(self):
        self._link_gtr_funds = None
        self._link_gtr_funds_api = None
        self._project_to_funds = None
        self._gtr_funds = None

    def get_project_funds(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """
        Adds funding amount and dates to the projects.

        NB: For a single project there are several funding entries, often with duplicated information,
        or amounts that don't correspond to the amounts on gtr.ukri.org. Therefore, while this method
        might still be valid to get project end dates, for safer funding amount data, use get_project_funds_api()

        The following steps are executed:
          - Fetching the funding table
          - Fetching the table linking project ids and funding ids
          - Linking GtR project ids to funding data using the funding and links tables

        Running it for the first time might take a while, as it needs to load funding data

        Args:
            gtr_projects: Data frame that must have a column "project_id"
            keep_only_dates: Keeps only the beginning and end dates

        Returns:
            Same input data frame with the following extra columns:
                - id: Funding id
                - fund_end: End of funding period
                - fund_start: Start of funding period
                - amount: Amount of funding
                - currencyCode: all values are 'GBP'

        """
        return (
            # Add funding ids to the projects table
            gtr_projects.merge(self.link_gtr_funds, on="project_id", how="left")
            # Add funding amounts and dates to the table
            .merge(self.gtr_funds, on="id", how="left")
            .query("category=='INCOME_ACTUAL'")
            .drop(["rel", "table_name"], axis=1)
            .rename(
                columns={"start_x": "start", "start_y": "fund_start", "end": "fund_end"}
            )
            .astype({"fund_start": "datetime64[ns]", "fund_end": "datetime64[ns]"})
        )

    def get_project_funds_api(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """Adds funding data to the projects (data that has been retrieved directly via API)"""
        return gtr_projects.merge(
            self.link_gtr_funds_api, on="project_id", how="left", validate="many_to_one"
        )

    def get_start_end_dates(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """
        Gets earliest start and latest end dates of project funds

        Args:
            gtr_projects: Data frame that must have a column "project_id"

        Returns:
            Same input data frame with the following extra columns:
                - fund_start: Start of funding period
                - fund_end: End of funding period
        """
        # Get project fund data
        projects_with_funds = self.get_project_funds(gtr_projects)

        # Get earliest values of "fund_start" for each project
        earliest_start_data = (
            projects_with_funds.groupby(["project_id", "fund_start"], sort=True)
            .count()
            .reset_index()
            .drop_duplicates("project_id", keep="first")[["project_id", "fund_start"]]
        )
        # Get earliest values of "fund_start" for each project
        latest_end_data = (
            projects_with_funds.groupby(["project_id", "fund_end"], sort=True)
            .count()
            .reset_index()
            .drop_duplicates("project_id", keep="last")[["project_id", "fund_end"]]
        )
        # Add project start and end dates to the input data frame
        return gtr_projects.merge(
            earliest_start_data, on="project_id", how="left", validate="many_to_one"
        ).merge(latest_end_data, on="project_id", how="left", validate="many_to_one")

    def get_funding_data(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """Adds reliable funding amount data, and funding start and end dates to the projects."""
        return pipe(gtr_projects, self.get_project_funds_api, self.get_start_end_dates)

    @property
    def gtr_funds(self):
        """GtR funding table"""
        if self._gtr_funds is None:
            self._gtr_funds = gtr.get_gtr_funds()
        return self._gtr_funds

    @property
    def link_gtr_funds(self):
        """Links between project ids and funding ids"""
        if self._link_gtr_funds is None:
            self._link_gtr_funds = gtr.get_link_table("gtr_funds")
        return self._link_gtr_funds

    @property
    def link_gtr_funds_api(self):
        """Links between project ids and funding ids, retrieved using API calls"""
        if self._link_gtr_funds_api is None:
            self._link_gtr_funds_api = gtr.get_gtr_funds_api()
        return self._link_gtr_funds_api


class CrunchbaseWrangler:
    """
    This class helps linking CB companies to other data such as the
    investment rounds (deals) and investors participating in the deals
    """

    def __init__(self):
        # Tables from the database
        self._cb_organisations = None
        self._cb_funding_rounds = None
        self._cb_investments = None
        self._cb_investors = None
        self._cb_category_groups = None
        self._cb_organisation_categories = None
        # Organisation categories (industries)
        self._industries = None
        self._industry_groups = None
        self._industry_to_group = None
        self._group_to_industries = None

    @property
    @lru_cache(1)
    def cb_organisations(self):
        """Full table of companies (this might take a minute to load in)"""
        if self._cb_organisations is None:
            self._cb_organisations = cb.get_crunchbase_orgs()
        return self._cb_organisations

    @property
    def cb_funding_rounds(self):
        """Table with investment rounds (all the deals for all companies)"""
        if self._cb_funding_rounds is None:
            self._cb_funding_rounds = cb.get_crunchbase_funding_rounds().drop(
                "index", axis=1
            )
        return self._cb_funding_rounds

    @property
    def cb_investments(self):
        """
        Table with investments for each round (deal). Note that one deal can
        have several investments pertaining to different investors
        """
        if self._cb_investments is None:
            self._cb_investments = cb.get_crunchbase_investments()
        return self._cb_investments

    @property
    def cb_investors(self):
        """Table with investors"""
        if self._cb_investors is None:
            self._cb_investors = cb.get_crunchbase_investors()
        return self._cb_investors

    @property
    def cb_category_groups(self):
        """Table of company categories (also called 'industries') and their broader 'category groups'"""
        if self._cb_category_groups is None:
            self._cb_category_groups = (
                # Load the dataframe
                cb.get_crunchbase_category_groups()
                # Convert to lower case
                .assign(
                    name=lambda x: x.name.str.lower(),
                    category_groups_list=lambda x: x.category_groups_list.str.lower(),
                )
                # Convert comma seperated category groups into a list
                .assign(
                    category_groups=lambda x: x.category_groups_list.apply(
                        split_comma_seperated_string
                    )
                )
                # Harmonise the naming of categories/industries column
                .rename(
                    columns={
                        "name": "industry",
                        "category_groups_list": "industry_groups_list",
                        "category_groups": "industry_groups",
                    }
                )
            )
        return self._cb_category_groups

    @property
    def cb_organisation_categories(self):
        """Table of companies and their corresponing categories (industries)"""
        if self._cb_organisation_categories is None:
            self._cb_organisation_categories = (
                cb.get_crunchbase_organizations_categories()
                # Harmonise the naming of categories/industries column
                .rename(columns={"category_name": "industry"})
            )
        return self._cb_organisation_categories

    @property
    def industries(self):
        """A list of the 700+ categories (industries)"""
        if self._industries is None:
            self._industries = (
                self.cb_category_groups["industry"].sort_values().to_list()
            )
        return self._industries

    @property
    def industry_groups(self):
        """A list of the broader industry (category) groups"""
        if self._industry_groups is None:
            self._industry_groups = sorted(
                list(
                    set(
                        itertools.chain(
                            *self.cb_category_groups["industry_groups"].to_list()
                        )
                    )
                )
            )
        return self._industry_groups

    @property
    def industry_to_group(self):
        if self._industry_to_group is None:
            self._industry_to_group = dict(
                zip(
                    self.cb_category_groups.industry,
                    self.cb_category_groups.industry_groups.to_list(),
                )
            )
        return self._industry_to_group

    @property
    def group_to_industries(self):
        if self._group_to_industries is None:
            df = (
                self.cb_category_groups.explode("industry_groups")
                .groupby("industry_groups")
                .agg(industry=("industry", lambda x: x.tolist()))
                .reset_index()
            )
            self._group_to_industries = dict(
                zip(df.industry_groups, df.industry.to_list())
            )
        return self._group_to_industries

    def get_funding_rounds(
        self, cb_organisations: pd.DataFrame, org_id_column: str = "id"
    ) -> pd.DataFrame:
        """
        Add funding round information to a table with CB organisations

        Args:
            cb_organisations: Data frame that must have a columns with crunchbase
                organisation ids and names

        Returns:
            Dataframe with organisations specified by 'org_id' and 'name', and
            all their funding rounds (deals). Some of the important columns include:
                'funding_round_id': unique round id
                'announced_on': date of the round
                'investment_type': the type of round (seed, series etc.)
                'post_money_valuation_usd': valuation of the company
                'raised_amount_usd': amount of money raised in the deal

        """
        return (
            # Keep only company name and id
            cb_organisations[[org_id_column, "name"]]
            # Harmonise column names and avoid clashes
            .rename(columns={org_id_column: "org_id"})
            # Add funding round data
            .merge(
                self.cb_funding_rounds.drop("name", axis=1),
                on="org_id",
            )
            # More informative column names
            .rename(columns={"id": "funding_round_id"})
            .sort_values("announced_on")
            .assign(
                # Convert to thousands
                raised_amount=lambda x: x["raised_amount"] / 1000,
                raised_amount_usd=lambda x: x["raised_amount_usd"] / 1000,
                # Convert date strings to datetimes
                announced_on_date=lambda x: pd.to_datetime(x["announced_on"]),
            )
            # Get years from dates
            .assign(year=lambda x: get_years(x["announced_on_date"]))
            # Convert USD currency to GBP
            .pipe(self.convert_deal_currency_to_gbp)
        )

    @staticmethod
    def convert_deal_currency_to_gbp(
        funding: pd.DataFrame,
        date_column: str = "announced_on_date",
        amount_column: str = "raised_amount",
        usd_amount_column: str = "raised_amount_usd",
        converted_column: str = "raised_amount_gbp",
    ) -> pd.DataFrame:
        """
        Convert USD currency to GBP using CurrencyConverter package.
        Deal dates should be provided in the datetime.date format
        NB: Rate conversion for dates before year 2000 is not reliable and hence
        is not carried out (the function returns nulls instead)

        Args:
            funding: A dataframe which must have a column for a date and amount to be converted
            date_column: Name of column with deal dates
            amount_column: Name of column with the amounts in the original currency
            amount_column_usd: Name of column with the amounts in USD
            converted_column: Name for new column with the converted amounts

        Returns:
            Same dataframe with an extra column for the converted amount

        """
        # Check if there is anything to convert
        rounds_with_funding = len(funding[-funding[usd_amount_column].isnull()])
        df = funding.copy()
        if rounds_with_funding > 0:
            # Set up the currency converter
            Converter = CurrencyConverter(
                fallback_on_missing_rate=True,
                fallback_on_missing_rate_method="linear_interpolation",
                # If the date is out of bounds (eg, too recent)
                # then use the closest date available
                fallback_on_wrong_date=True,
            )
            # Convert currencies
            converted_amounts = []
            for _, row in df.iterrows():
                # Only convert deals after year 1999
                if row[date_column].year >= 2000:
                    converted_amounts.append(
                        Converter.convert(
                            row[usd_amount_column], "USD", "GBP", date=row[date_column]
                        )
                    )
                else:
                    converted_amounts.append(np.nan)
            df[converted_column] = converted_amounts
            # For deals that were originally in GBP, use the database values
            deals_in_gbp = df["raised_amount_currency_code"] == "GBP"
            df.loc[deals_in_gbp, converted_column] = df.loc[
                deals_in_gbp, amount_column
            ].copy()
        else:
            # If nothing to convert, copy the nulls and return
            df[converted_column] = df[amount_column].copy()
        return df

    def get_funding_round_investors(self, funding_rounds: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the investors involved in the specified funding rounds

        Args:
            Dataframe with organisations specified by 'org_id' and 'name', and
            all their funding rounds (deals). Some of the important columns include:

        Returns:
            Dataframe with extra columns specifying investment round details:
                'funding_round_name': Name of the funding round
                'investor_name': Name of the investor organisation
                'investor_id' Crunchbase organisation identifier
                'investor_type': Specifies if investor is a person or an organisation
                'is_lead_investor': Specifies whether the organisation is leading the round (value is 1 in that case)

        """
        return funding_rounds.merge(
            (
                self.cb_investments[
                    [
                        "funding_round_id",
                        "funding_round_name",
                        "investor_name",
                        "id",
                        "investor_type",
                        "is_lead_investor",
                    ]
                ].rename(columns={"id": "investor_id"})
            ),
            on="funding_round_id",
            how="left",
        ).assign(
            raised_amount=lambda x: x.raised_amount.fillna(0),
            raised_amount_usd=lambda x: x.raised_amount_usd.fillna(0),
        )

    def get_organisation_investors(
        self, cb_organisations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Gets the investors and corresponding funding rounds for the specified organisations

        Args:
            cb_organisations: Data frame that must have a columns with crunchbase
                organisation ids and names

        Returns:
             Dataframe with extra columns specifying investors and the corresponding investment rounds
             (see docs for get_funding_rounds and get_funding_round_investors for more information)

        """
        return pipe(
            cb_organisations, self.get_funding_rounds, self.get_funding_round_investors
        )

    def get_companies_in_industries(
        self, industries_names: Iterator[str]
    ) -> pd.DataFrame:
        """
        Get companies belonging to the specified industries
        NB: Might take a minute when running for the first time, as it needs
        to load in the full table of companies

        Args:
            industries_names: A list of industry names; see all industries in CrunchbaseWrangler().industries

        Returns:
            A dataframe with organisation data
        """
        # Find ids of companies in the specified industries
        company_ids = self.cb_organisation_categories[
            self.cb_organisation_categories.industry.isin(industries_names)
        ].organization_id
        # Return data for organisatons that are in the specified industries
        return self.cb_organisations[self.cb_organisations.id.isin(company_ids)]

    def get_company_industries(
        self, cb_organisations: pd.DataFrame, return_lists: bool = False
    ) -> pd.DataFrame:
        """
        Get industries of the specified companies. Note that a company can be in more
        than one industry.

        Args:
            cb_organisations: Data frame that must have a columns with crunchbase
                organisation ids and name
            return_lists: If True, will return a row per company, and all industries
                for each company collected in a list

        Returns:
            Dataframe with organisation ids, names and their industries
        """

        company_industries = (
            cb_organisations[["id", "name"]]
            .merge(
                self.cb_organisation_categories[["organization_id", "industry"]],
                left_on="id",
                right_on="organization_id",
                how="left",
                validate="one_to_many",
            )
            .drop("organization_id", axis=1)
        )
        if return_lists:
            return (
                company_industries.groupby(["id"])
                .agg(industry=("industry", lambda x: x.tolist()))
                .merge(
                    cb_organisations[["id", "name"]],
                    on="id",
                    how="left",
                    validate="one_to_one",
                )
            )[["id", "name", "industry"]]
        else:
            return company_industries

    def select_companies_by_industries(
        self, cb_organisations: pd.DataFrame, filtering_industries: Iterator[str]
    ) -> pd.DataFrame:
        """
        From an initial set of companies, select a subset that belogs to any of the specified filtering_industries


        Args:
            cb_organisations: Data frame that must have a columns with crunchbase
                organisation ids and name
            filtering_industries: A list of industry names

        Returns:
            Dataframe with organisation data and their industries

        """
        # Get industries for each company
        company_industries = self.get_company_industries(
            cb_organisations, return_lists=True
        )
        # Check if the industry lists contain the specified filtering industries
        is_in_filtering_industries = company_industries["industry"].apply(
            lambda x: is_string_in_list(filtering_industries, x)
        )
        # Get ids of the companies to keep
        filtered_ids = company_industries[is_in_filtering_industries].id.to_list()
        return (
            cb_organisations[cb_organisations.id.isin(filtered_ids)]
            # Add industries list
            .merge(
                company_industries, on=["id", "name"], how="left", validate="one_to_one"
            )
        )


def get_years(dates: Iterator[datetime.date]) -> Iterator:
    """Converts a list of datetimes to years"""
    return [x.year for x in dates]


def split_comma_seperated_string(text: str) -> Iterator[str]:
    return [s.strip() for s in text.split(",")] if type(text) is str else []


def is_string_in_list(list_of_strings: Iterator[str], list_to_check: Iterator[str]):
    """Checks if any of the provided strings in list_of_strings are in the specified list_to_check"""
    return True in [s in list_to_check for s in list_of_strings]
