import pandas as pd
from toolz import pipe
import innovation_sweet_spots.getters.gtr as gtr


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
            self.link_gtr_funds_api, on="project_id", how="left", validate="one_to_one"
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
            earliest_start_data, on="project_id", how="left", validate="one_to_one"
        ).merge(latest_end_data, on="project_id", how="left", validate="one_to_one")

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
