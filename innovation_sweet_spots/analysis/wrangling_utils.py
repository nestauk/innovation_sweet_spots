import pandas as pd
import numpy as np
from toolz import pipe
from currency_converter import CurrencyConverter
from datetime import datetime
from typing import Iterator
import itertools
from innovation_sweet_spots import logging
import innovation_sweet_spots.getters.gtr as gtr
import innovation_sweet_spots.getters.crunchbase as cb
import innovation_sweet_spots.getters.dealroom as dealroom


class GtrWrangler:
    """
    This class helps linking GtR projects to other GtR data such as the
    project funding or the participating organisations
    """

    def __init__(self):
        self._gtr_projects = None
        # Funding
        self._link_gtr_funds = None
        self._link_gtr_funds_api = None
        self._project_to_funds = None
        self._gtr_funds = None
        # Organisations
        self._gtr_organisations = None
        self._gtr_organisations_locations = None
        self._link_gtr_organisations = None
        # People
        self._gtr_persons = None
        self._link_gtr_persons = None
        # Research topics
        self._link_gtr_topics = None
        self._gtr_topics = None

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
            .query(
                "category==['INCOME_ACTUAL', 'EXPENDITURE_ACTUAL', 'EXPENDITURE_PROJECTED']"
            )
            .drop(["rel", "table_name"], axis=1)
            .rename(
                columns={"start_x": "start", "start_y": "fund_start", "end": "fund_end"}
            )
            .astype({"fund_start": "datetime64[ns]", "fund_end": "datetime64[ns]"})
        )

    def get_project_funds_api(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """Adds funding data to the projects (data that has been retrieved directly via API)"""
        return gtr_projects.merge(
            self.link_gtr_funds_api,
            on="project_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_drop"),
        ).filter(regex="^(?!.*_drop)")

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
        return (
            gtr_projects.merge(
                earliest_start_data, on="project_id", how="left", validate="many_to_one"
            )
            .merge(latest_end_data, on="project_id", how="left", validate="many_to_one")
            .astype({"fund_start": "datetime64[ns]", "fund_end": "datetime64[ns]"})
        )

    def get_funding_data(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """Adds reliable funding amount data, and funding start and end dates to the projects."""
        return pipe(gtr_projects, self.get_project_funds_api, self.get_start_end_dates)

    def split_funding_data(
        self, gtr_projects: pd.DataFrame, time_period: str = "month"
    ) -> pd.DataFrame:
        """
        Splits GtR funding evenly over the duration of the projects

        Args:
            gtr_projects: Dataframe that must have columns for 'fund_start'
                and 'fund_end'
            time_period: Time period to split the funding data across,
                must be one of 'year', 'month', 'quarter'. Defaults to 'month'.

        Returns:
            Same input dataframe but with additional rows for
            the time periods that the funding has been split across
        """
        # Check time period is valid
        check_valid(time_period, ["year", "month", "quarter"])

        # Add split info
        frequency = time_period[0].capitalize()
        gtr_projects["date_range"] = gtr_projects.apply(
            lambda x: pd.period_range(
                start=x.fund_start, end=x.fund_end, freq=frequency
            ).to_timestamp(),
            axis=1,
        )
        gtr_projects["amount_per_period"] = gtr_projects.amount / gtr_projects.apply(
            lambda x: len(x.date_range), axis=1
        )

        # Create funding data split
        funding_data_split = []
        exclude_cols = ["amount_per_period", "date_range", "fund_start", "fund_end"]
        for _, row in gtr_projects.iterrows():
            fd_split = {
                col: row[col]
                for col in [
                    col for col in gtr_projects.columns if col not in exclude_cols
                ]
            }
            fd_split["amount"] = row["amount_per_period"]
            fd_split["start"] = row["date_range"]
            funding_data_split.append(pd.DataFrame(data=fd_split))
        return pd.concat(funding_data_split).reset_index(drop=True)

    def get_project_organisations(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """
        Adds participating organisations to the projects.

        Args:
            gtr_projects: Data frame that must have a column "project_id"

        Returns:
            Same input data frame with the following extra columns:
                - id: Organisation id
                - organisation_relation: Indicates different types of participation
                - organisation_name: Name of the organisation

        """
        return (
            # Add organisation ids to the projects table
            gtr_projects.merge(self.link_gtr_organisations, on="project_id", how="left")
            # Add organisation data (NB: Ignoring addresses column)
            .merge(self.gtr_organisations[["id", "name"]], on="id", how="left")
            .drop(["table_name"], axis=1)
            .rename(
                columns={"name": "organisation_name", "rel": "organisation_relation"}
            )
        )

    def get_organisation_locations(
        self, gtr_organisations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds latitude, longitude, countries and addresses of participating organisations

        Args:
            gtr_organisations: Data frame that must have an organisation "id" column

        Returns:
            Same input data frame with the following extra columns:
                - address
                - country_name
                - country_alpha_2: 2-letter country code
                - continent
                - latitude
                - longitude
            Also available, but not returned:
                - country_alpha_3: 3-letter country code
                - country_numeric: Presumably ISO 3166 country code
        """
        return (
            # Add addresses
            gtr_organisations.merge(
                self.gtr_organisations[["id", "addresses"]], on="id", how="left"
            )
            # Add country and lat/long
            .merge(self.gtr_organisations_locations, on="id", how="left").drop(
                ["country_alpha_3", "country_numeric"], axis=1
            )
        )

    def get_organisations_and_locations(
        self, gtr_projects: pd.DataFrame
    ) -> pd.DataFrame:
        """Adds organisations and their location data"""
        return pipe(
            gtr_projects,
            self.get_project_organisations,
            self.get_organisation_locations,
        )

    def get_persons(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """
        Adds participating persons to the projects.

        Args:
            gtr_projects: Data frame that must have a column "project_id"

        Returns:
            Same input data frame with the following extra columns:
                - id: Person id
                - person_relation: Indicates different types of participation
                - firstName
                - otherNames
                - surname

        """
        return (
            # Add person ids to the projects table
            gtr_projects.merge(self.link_gtr_persons, on="project_id", how="left")
            # Add person data
            .merge(self.gtr_persons, on="id", how="left")
            .drop(["table_name"], axis=1)
            .rename(columns={"rel": "person_relation"})
        )

    def get_research_topics(self, gtr_projects: pd.DataFrame) -> pd.DataFrame:
        """
         Add GtR research topics to projects. Note that about half of the projects are 'Unclassified'

         Args:
             gtr_projects: Data frame that must have a column "project_id"

         Returns:
             Same input data frame with the following extra columns:
                 - topic: 750+ different project categories
                 - topic_type: One of the following: 'researchActivity', 'researchTopic', 'researchSubject',
        'healthCategory', 'rcukProgramme'
        """
        return (
            gtr_projects.merge(self.link_gtr_topics, on="project_id", how="left")
            .merge(self.gtr_topics, on="id", how="left")
            .drop(["rel", "table_name", "id"], axis=1)
        )

    def get_projects_in_research_topics(
        self, research_topics: Iterator[str]
    ) -> pd.DataFrame:
        """
        Get projects with the specified GtR research topics

        Args:
            research_topics: A list of research topics; see all topics in GtrWrangler().gtr_topics

        Returns:
            A dataframe with project data
        """
        return self.get_research_topics(self.gtr_projects).query(
            "topic in @research_topics"
        )

    def add_project_data(
        self,
        dataframe: pd.DataFrame,
        id_column: str = "project_id",
        columns: Iterator[str] = None,
    ) -> pd.DataFrame:
        """Adds basic GtR project data such as titles and abstracts to a dataframe with project ids

        Args:
            dataframe: Dataframe with project ids
            id_column: Name of the project id column in the input dataframe
            columns: Columns of project data to add; by default will add all project table columns

        Returns:
            Input dataframe with extra columns with project data
        """
        # By default, select all columns from the projects table
        if columns is None:
            columns = self.gtr_projects.columns
        # Ensure that project_id is in the columns
        columns = {"project_id"} | set(columns)
        return dataframe.merge(
            self.gtr_projects[columns],
            left_on=id_column,
            right_on="project_id",
            how="left",
        )

    @property
    def gtr_projects(self):
        """GtR projects table"""
        if self._gtr_projects is None:
            self._gtr_projects = gtr.get_gtr_projects()
        return self._gtr_projects

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

    @property
    def gtr_organisations(self):
        """Organisations participating in research projects"""
        if self._gtr_organisations is None:
            self._gtr_organisations = gtr.get_gtr_organisations()
        return self._gtr_organisations

    @property
    def gtr_organisations_locations(self):
        """Organisation locations"""
        if self._gtr_organisations_locations is None:
            self._gtr_organisations_locations = gtr.get_gtr_organisations_locations()
        return self._gtr_organisations_locations

    @property
    def link_gtr_organisations(self):
        """Links between project ids and organisation ids"""
        if self._link_gtr_organisations is None:
            self._link_gtr_organisations = gtr.get_link_table("gtr_organisations")
        return self._link_gtr_organisations

    @property
    def gtr_persons(self):
        """Links between project ids and organisation ids"""
        if self._gtr_persons is None:
            self._gtr_persons = gtr.get_gtr_persons()
        return self._gtr_persons

    @property
    def link_gtr_persons(self):
        """Links between project ids and organisation ids"""
        if self._link_gtr_persons is None:
            self._link_gtr_persons = gtr.get_link_table("gtr_persons")
        return self._link_gtr_persons

    @property
    def link_gtr_topics(self):
        """Links between project ids and research topic ids"""
        if self._link_gtr_topics is None:
            self._link_gtr_topics = gtr.get_link_table("gtr_topic")
        return self._link_gtr_topics

    @property
    def gtr_topics(self):
        """GtR research topics"""
        if self._gtr_topics is None:
            self._gtr_topics = gtr.get_gtr_topics().rename(columns={"text": "topic"})
        return self._gtr_topics

    @property
    def gtr_topics_list(self):
        """Returns a sorted list of GtR research topics (categories)"""
        return sorted(self.gtr_topics.topic.to_list())


class CrunchbaseWrangler:
    """
    This class helps linking CB companies to other data such as the
    investment rounds (deals) and investors participating in the deals
    """

    def __init__(self, cb_data_path: str = None):
        """
        Sets up the CrunchbaseWrangler class

        Args:
            cb_data_path: Optional argument to specify the location of the
                Crunchbase data, if it is different from the default location;
                NB: Note that this changes a global variable.
        """
        # Tables from the database
        self._cb_organisations = None
        self._cb_funding_rounds = None
        self._cb_investments = None
        self._cb_investors = None
        self._cb_category_groups = None
        self._cb_organisation_categories = None
        self._cb_people = None
        self._cb_degrees = None
        # Organisation categories (industries)
        self._industries = None
        self._industry_groups = None
        self._industry_to_group = None
        self._group_to_industries = None
        # Set data path (optional)
        if cb_data_path is not None:
            cb.CB_PATH = cb_data_path

    @property
    def cb_organisations(self):
        """Full table of companies (this might take a minute to load in)"""
        if self._cb_organisations is None:
            self._cb_organisations = cb.get_crunchbase_orgs()
        return self._cb_organisations

    @property
    def cb_funding_rounds(self):
        """Table with investment rounds (all the deals for all companies)"""
        if self._cb_funding_rounds is None:
            self._cb_funding_rounds = cb.get_crunchbase_funding_rounds()
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
    def cb_people(self):
        """Table with people"""
        if self._cb_people is None:
            self._cb_people = cb.get_crunchbase_people()
        return self._cb_people

    @property
    def cb_degrees(self):
        """Table with degrees"""
        if self._cb_degrees is None:
            self._cb_degrees = cb.get_crunchbase_degrees()
        return self._cb_degrees

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
        """Dictionary that maps narrow Crunchbase industries (categories) to broader groups"""
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
        """Dictionary that maps from broader industry group to a set of narrower industries (categories)"""
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
                'raised_amount_usd': amount of money raised in the deal (in thousands)

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
            all their funding rounds (deals).

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

    def get_company_persons(self, cb_organisations: pd.DataFrame) -> pd.DataFrame:
        """
        Adds people associated with the specified companies.

        Args:
            cb_organisations: Data frame that must have columns with crunchbase
                organisation ids and name

        Returns:
            Dataframe with organisation ids, names and their person data
        """
        return (
            cb_organisations[["id", "name"]]
            .merge(
                self.cb_people,
                left_on="id",
                right_on="featured_job_organization_id",
                how="inner",
            )
            .rename(
                columns={
                    "id_x": "id",
                    "id_y": "person_id",
                    "name_x": "name",
                    "name_y": "person_name",
                }
            )
        )

    def get_person_degrees(self, cb_persons: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the university degrees for the specified persons.

        Args:
            cb_persons: Data frame that must have a column 'person id' with
                crunchbase person ids

        Returns:
            Dataframe with person degree data
        """
        return (
            cb_persons[["person_id", "person_name"]]
            .merge(
                self.cb_degrees,
                on="person_id",
                how="left",
            )
            .drop("person_name_y", axis=1)
            .rename(
                columns={
                    "name_x": "name",
                    "name_y": "degree_name",
                    "person_name_x": "person_name",
                }
            )
        )

    def get_company_education_data(
        self,
        cb_organisations: pd.DataFrame,
        columns=["institution_name", "degree_type", "subject", "completed_on"],
    ) -> pd.DataFrame:
        """
        Gets the employee education data for the specified companies.
        By default, it will fetch the names of institutions, degrees and completion date.

        Args:
            cb_organisations: Data frame that must have columns with crunchbase
                organisation ids and name

        Returns:
            Dataframe with organisation id, name and education data of their employees
        """
        # Look up persons
        persons = self.get_company_persons(cb_organisations)
        return (
            # Get peoples degrees
            persons.merge(
                self.get_person_degrees(persons)[["person_id"] + columns],
                on="person_id",
                how="left"
                # Clean up columns
            )[["id", "name"] + columns]
            # Remove rows corresponding to persons with no education data
            .dropna(subset=columns, how="all")
        )

    def add_company_data(
        self,
        dataframe: pd.DataFrame,
        id_column: str = "id",
        columns: Iterator[str] = None,
    ) -> pd.DataFrame:
        """Adds basic company data such as name and description to a dataframe with project ids

        Args:
            dataframe: Dataframe with project ids
            id_column: Name of the project id column in the input dataframe
            columns: Columns of project data to add; by default will add all project table columns

        Returns:
            Input dataframe with extra columns with project data
        """
        # By default, select all columns from the projects table
        if columns is None:
            columns = self.cb_organisations.columns
        # Ensure that id is in the columns
        columns = {"id"} | set(columns)
        return dataframe.merge(
            self.cb_organisations[columns],
            left_on=id_column,
            right_on="id",
            how="left",
        )


HARMONISED_DEALROOM_COL_NAMES = {
    "HQ COUNTRY": "country",
    "HQ CITY": "city",
}


class DealroomWrangler:
    """
    This class helps exploring data downloaded from the Dealroom business platform
    """

    def __init__(self, dataset="foodtech"):
        self._company_data = None
        self._foodtech_data = None
        # Leave the option for other data downloads in the future
        if dataset == "foodtech":
            self._company_data = self.foodtech_data
        # Wrangled descriptors
        self._company_tags = None
        self._company_industries = None
        self._company_subindustries = None
        self._company_labels = None
        self._labels = None
        self._funding_rounds = None

    @property
    def company_data(self):
        """Company data"""
        return self._company_data

    @property
    def foodtech_data(self) -> pd.DataFrame:
        """
        Company data downloaded from Dealroom platform, for the
        purposes of the Innovation Sweet Spots project on food tech
        """
        if self._foodtech_data is None:
            self._foodtech_data = (
                dealroom.get_foodtech_companies()
                # Deduplicate IDs
                .astype({"id": str})
                .drop_duplicates("id", keep="first")
                .pipe(self.process_input_data)
            )
        return self._foodtech_data

    def harmonise_column_names(self, df: pd.DataFrame):
        """"""
        return df.rename(columns=HARMONISED_DEALROOM_COL_NAMES)

    def process_input_data(self, df: pd.DataFrame):
        """Initial processing"""
        return (
            df
            # Remove spurious entries
            .query("id != 'Error retrieving row data'")
            # Normalise the launch date (NB: only year is taken into account for now)
            .assign(
                founded_on=lambda df: pd.to_datetime(
                    df["LAUNCH DATE"].apply(self.extract_year)
                )
            ).pipe(self.harmonise_column_names)
        )

    @staticmethod
    def extract_year(date: str):
        """Extracts year from the 'LAUNCH DATE' column"""
        if type(date) is str:
            return date[0:4]
        else:
            return np.nan

    @property
    def company_tags(self) -> pd.DataFrame:
        """
        Returns table with company id numbers and tags
        """
        if self._company_tags is None:
            self._company_tags = self.explode_dealroom_table("TAGS")
        return self._company_tags

    @property
    def company_industries(self) -> pd.DataFrame:
        """
        Returns table with company id numbers and industries
        """
        if self._company_industries is None:
            self._company_industries = self.explode_dealroom_table("INDUSTRIES")
        return self._company_industries

    @property
    def company_subindustries(self) -> pd.DataFrame:
        """
        Returns table with company id numbers and sub-industries
        """
        if self._company_subindustries is None:
            self._company_subindustries = self.explode_dealroom_table("SUB INDUSTRIES")
        return self._company_subindustries

    @property
    def funding_rounds(self) -> pd.DataFrame:
        if self._funding_rounds is None:
            self._funding_rounds = self.get_funding_rounds()
        return self._funding_rounds

    def explode_dealroom_table(self, column_name: str) -> pd.DataFrame:
        """Returns table with company ids and and a row for each separate element of the specified column"""
        return explode_table(self.company_data[["id", column_name]], column_name, ";")

    @staticmethod
    def get_years_from_parentheses(column_name: str) -> Iterator[int]:
        """
        Converts a column name of format 'COLUMN_NAME (year_1, year_2)'
        to 'year_1;year_2'
        """
        return ";".join(column_name.split("(")[-1].split(")")[0].split(","))

    @staticmethod
    def get_descriptor_name(column_name: str) -> str:
        """
        Converts a column name of format 'COLUMN_NAME (year_1, year_2)'
        to a string 'COLUMN_NAME'
        """
        return column_name.split("(")[0].strip()

    def get_ids_in_subindustry(self, subindustry: str):
        """Get company ids in a subindustry"""
        return self.company_subindustries.query(
            "`SUB INDUSTRIES` == @subindustry"
        ).id.to_list()

    def get_companies_by_subindustry(self, subindustry: str):
        """Get companies that are in the specific subindustry"""
        ids_in_industry = self.get_ids_in_subindustry(subindustry)
        return self.company_data.query("id in @ids_in_industry").assign(
            subindustry=subindustry
        )

    def get_rounds_by_subindustry(self, subindustry: str):
        """Get investment rounds for companies in the specific subindustry"""
        ids_in_industry = self.get_ids_in_subindustry(subindustry)
        return self.funding_rounds.query("id in @ids_in_industry").assign(
            subindustry=subindustry
        )

    def get_ids_in_industry(self, industry: str):
        """Get company ids in a subindustry"""
        return self.company_industries.query("`INDUSTRIES` == @industry").id.to_list()

    def get_companies_by_industry(self, industry: str):
        """Get companies that are in the specific subindustry"""
        ids_in_industry = self.get_ids_in_industry(industry)
        return self.company_data.query("id in @ids_in_industry").assign(
            industry=industry
        )

    @property
    def company_labels(self):
        """Companies and all their industry, sub-industry and tag labels"""
        if self._company_labels is None:
            self._company_labels = self.get_all_company_labels()
        return self._company_labels

    @property
    def labels(self):
        """All unique industry, sub-industry and tag labels"""
        if self._labels is None:
            self._labels = (
                self.company_labels.drop_duplicates("Category")
                .sort_values("Category")
                .drop("id", axis=1)
                .reset_index(drop=True)
            )
        return self._labels

    def get_all_company_labels(self) -> pd.DataFrame:
        """Compiles all industry, sub-industry and tag labels into one dataframe"""
        label = "SUB INDUSTRIES"
        sub = self.company_subindustries.rename(columns={label: "Category"}).assign(
            label_type=label
        )

        label = "INDUSTRIES"
        ind = self.company_industries.rename(columns={label: "Category"}).assign(
            label_type=label
        )

        label = "TAGS"
        tags = self.company_tags.rename(columns={label: "Category"}).assign(
            label_type=label
        )

        company_labels = pd.concat([sub, ind, tags], ignore_index=True)
        company_labels = company_labels[-company_labels.Category.isnull()]
        return company_labels

    def get_rounds_by_industry(self, industry: str):
        """Get investment rounds for companies in the specific subindustry"""
        ids_in_industry = self.get_ids_in_industry(industry)
        return self.funding_rounds.query("id in @ids_in_industry").assign(
            industry=industry
        )

    def explode_timeseries(self, column_name: str) -> pd.DataFrame:
        """
        Explodes columns with yearly time series, eg "PROFIT (year_1, year_2, ...)" into a
        dataframe with a row for each year
        """
        return (
            self.company_data[["id", column_name]]
            .assign(
                new_column=lambda x: x[column_name].apply(
                    lambda y: split_comma_seperated_string(y, ";")
                ),
                year=str(self.get_years_from_parentheses(column_name)),
            )
            .assign(
                year=lambda x: x.year.apply(
                    lambda y: split_comma_seperated_string(y, ";")
                )
            )
            .explode(["new_column", "year"])
            .drop(column_name, axis=1)
            .drop_duplicates()
            .reset_index(drop=True)
            .rename(columns={"new_column": self.get_descriptor_name(column_name)})
        )

    def impute_month(self, date: str):
        """Imputes month, eg 2012 becomes jan/2012"""
        # If only year is present
        if date.isnumeric() and (len(date) == 4):
            return f"jan/{date}"
        else:
            return date

    def get_funding_rounds(self, currencies=["GBP", "USD"]) -> pd.DataFrame:
        """
        Returns funding rounds for companies

        Args:
            currencies: List of currencies in which to convert all funding round amounts
        """
        # Get the columns related to investment time series
        company_funding_data = (
            self.company_data[
                ["id"] + dealroom.COLUMN_CATEGORIES["Funding (detailed)"]
            ].fillna("n/a")
        ).copy()
        # Remove companies without any funding data
        company_funding_data = company_funding_data[
            company_funding_data["EACH ROUND DATE"] != "n/a"
        ]
        # Turn the separated entries into lists
        for col in dealroom.COLUMN_CATEGORIES["Funding (detailed)"]:
            company_funding_data[col] = company_funding_data[col].apply(
                lambda x: split_comma_seperated_string(x, ";")
            )

        # Check where there is an issue with the investor column (small number of rounds)
        df_lengths = pd.DataFrame()
        for col in dealroom.COLUMN_CATEGORIES["Funding (detailed)"]:
            df_lengths[col] = company_funding_data[col].apply(lambda x: len(x))
        # NB: The assumption is that only the 'EACH ROUND INVESTORS' column is an issue
        ambiguous_idx = df_lengths[
            df_lengths["EACH ROUND TYPE"] != df_lengths["EACH ROUND INVESTORS"]
        ].index
        # For the ambigouous cases, replace investors with n/a, and explode funding rounds
        ambiguous_rounds = (
            company_funding_data.loc[ambiguous_idx]
            .copy()
            .assign(null_investors="n/a")
            .drop("EACH ROUND INVESTORS", axis=1)
            .rename(columns={"null_investors": "EACH ROUND INVESTORS"})
            .explode(
                [
                    "EACH ROUND TYPE",
                    "EACH ROUND AMOUNT",
                    "EACH ROUND CURRENCY",
                    "EACH ROUND DATE",
                ]
            )
        )

        # Explode funding rounds for non-ambiguous cases
        company_funding_rounds = company_funding_data.drop(ambiguous_idx).explode(
            dealroom.COLUMN_CATEGORIES["Funding (detailed)"]
        )
        # Combine both exploded dataframes
        company_funding_rounds = (
            pd.concat([company_funding_rounds, ambiguous_rounds], ignore_index=True)
            .reset_index(drop=True)
            .assign(
                **{
                    # Convert funding date to a datetime format
                    "announced_on": lambda df: pd.to_datetime(
                        df["EACH ROUND DATE"].apply(self.impute_month),
                        format="%b/%Y",
                        errors="coerce",
                    ),
                    # Create a temporary funding round id
                    "funding_round_id": lambda df: list(range(0, len(df))),
                }
            )
        )
        #         company_funding_rounds['fake_date'] = pd.to_datetime('2020-05-23')
        # Convert currencies to GBP and USD
        company_funding_rounds_converted = [
            convert_currency(
                funding=(
                    company_funding_rounds.query("`EACH ROUND AMOUNT` != 'n/a'").query(
                        "`EACH ROUND CURRENCY` != 'n/a'"
                    )
                ),
                date_column="announced_on",
                #                 date_column="fake_date",
                amount_column="EACH ROUND AMOUNT",
                currency_column="EACH ROUND CURRENCY",
                target_currency=curr,
            )
            for curr in currencies
        ]
        # Add the converted amounts to the original dataframe
        for i, df in enumerate(company_funding_rounds_converted):
            company_funding_rounds = (
                company_funding_rounds.merge(
                    df[["id", "funding_round_id", df.columns[-1]]],
                    on=["id", "funding_round_id"],
                    how="left",
                )
                .replace("n/a", np.nan)
                .rename(
                    columns={df.columns[-1]: f"raised_amount_{currencies[i].lower()}"}
                )
                .astype(
                    {
                        f"raised_amount_{currencies[i].lower()}": float,
                    }
                )
            )
        return company_funding_rounds


def get_years(dates: Iterator[datetime.date]) -> Iterator:
    """Converts a list of datetimes to years"""
    return [x.year for x in dates]


def split_comma_seperated_string(text: str, separator: str = ",") -> Iterator[str]:
    """Splits a string where commas are; for example: 'a, b' -> ['a', 'b']"""
    return [s.strip() for s in text.split(f"{separator}")] if type(text) is str else []


def is_string_in_list(list_of_strings: Iterator[str], list_to_check: Iterator[str]):
    """Checks if any of the provided strings in list_of_strings are in the specified list_to_check"""
    return True in [s in list_to_check for s in list_of_strings]


def check_valid(check_var, check_list):
    """Raise ValueError is check_var not in check_list"""
    if check_var not in check_list:
        raise ValueError(f"{check_var} is not valid, it must be one of {check_list}.")


def explode_table(
    df: pd.DataFrame, column_name: str, separator: str = ","
) -> pd.DataFrame:
    """
    Explodes the specified column and does some housekeeping (eg, deduplication). The column
    should contain strings, which in turn contain substrings separated by a character (eg, comma).

    Args:
        df: Table that has a column with list-like text strings (eg comma-separated words)
        column_name: The name of the column with list-like strings
        separator: Character used to separate the elements in the list-like strings

    Returns:
        Exploded dataframe with a separate row for each element of the list-like strings
    """
    return (
        df.assign(
            new_column=lambda x: x[column_name].apply(
                lambda y: split_comma_seperated_string(y, separator)
            )
        )
        .explode("new_column")
        .drop(column_name, axis=1)
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"new_column": column_name})
    )


def convert_currency(
    funding: pd.DataFrame,
    date_column: str,
    amount_column: str,
    currency_column: str,
    converted_column: str = None,
    target_currency: str = "GBP",
) -> pd.DataFrame:
    """
    Convert amount in any currency to a target currency using CurrencyConverter package.
    Deal dates should be provided in the datetime.date format
    NB: Rate conversion for dates before year 2000 is not reliable and hence
    is not carried out (the function returns nulls instead)

    Args:
        funding: A dataframe which must have a column for a date and amount to be converted
        date_column: Name of column with deal dates
        amount_column: Name of column with the amounts in the original currency
        currency_column: Name of column with the currency codes (eg, 'USD', 'EUR' etc)
        converted_column: Name for new column with the converted amounts

    Returns:
        Same dataframe with an extra column for the converted amount

    """
    # Column name
    converted_column = (
        f"{amount_column}_{target_currency}"
        if converted_column is None
        else converted_column
    )
    # Check if there is anything to convert
    rounds_with_funding = len(funding[-funding[amount_column].isnull()])
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
            if (row[date_column].year >= 2000) and (
                row[currency_column] in Converter.currencies
            ):
                converted_amounts.append(
                    Converter.convert(
                        row[amount_column],
                        row[currency_column],
                        target_currency,
                        date=row[date_column],
                    )
                )
            else:
                converted_amounts.append(np.nan)
        df[converted_column] = converted_amounts
        # For deals that were originally in the target currency, use the database values
        deals_in_target_currency = df[currency_column] == target_currency
        df.loc[deals_in_target_currency, converted_column] = df.loc[
            deals_in_target_currency, amount_column
        ].copy()
    else:
        # If nothing to convert, copy the nulls and return
        df[converted_column] = df[amount_column].copy()
    return df
