"""
Script to create a csv file with Crunchbase organisations and R&D indicators
of their locations. The indicators are sourced from the Nesta/BEIS R&D dashboard:
https://access-research-development-spatial-data.beis.gov.uk/indicators

Usage:
python innovation_sweet_spots/pipeline/pilot/investment_predictions/create_dataset/create_beis_indicators.py --help
"""
from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.getters.crunchbase import get_crunchbase_orgs
from innovation_sweet_spots.utils.geo import get_crunchbase_nuts, add_nuts_to_crunchbase
from innovation_sweet_spots.getters.beis_indicators import get_beis_indicators
import pandas as pd
import typer

OUTPUT_FILEPATH = PROJECT_DIR / "outputs/finals/pilot_outputs/investment_predictions"
OUTPUT_FILENAME = "company_beis_indicators_{}.csv"


def add_indicator(
    organisations_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    indicator: str,
    window_end_date: int,
) -> pd.DataFrame:
    """
    Adds a column to the organisations_df with the most recent specified R&D indicator

    Args:
        organisations_df: Table with organisations and their NUTS2 regions
        indicators_df: Table with R&D indicators for different NUTS2 regions
        indicator: Indicator identifier name
        window_end_date: Cut-off year

    Returns:
        organisations_df table with additional column named after the specified indicator
    """
    # Find the specified indicator rows
    df_ind = indicators_df.query("indicator_id == @indicator")
    # Find the most recent allowed year
    max_year = df_ind.query("year < @window_end_date").year.max()
    df_ind = df_ind.query(f"year == @max_year")
    # Check if there are indicator values for the specified window
    if len(df_ind) != 0:
        # Log the indicator
        row = df_ind.iloc[0]
        logging.info(f"{indicator}, {row.title}, {row.year}")
        # Merge with the organisation table
        return (
            organisations_df.merge(
                df_ind[["region_id", "value"]],
                left_on=f"nuts2_{row.region_year_spec}",
                right_on="region_id",
                how="left",
            )
            .rename(columns={"value": indicator})
            .drop(["region_id"], axis=1)
            .fillna(-1)
        )
    else:
        # Return the input table if no values found
        logging.info((f"{indicator} NOT ADDED (no values for the specified window)"))
        return organisations_df


def create_beis_indicators(window_end_date: int = 2018, test: bool = False):
    """
    Creates a csv file with Crunchbase organisations and R&D indicators of their locations.

    Args:
        window_end_date: Cut-off year
        test: Flag for test mode
    """
    # Fetch companies
    nrows = (
        1000 if test else None
    )  # Hoping that there are a few UK companies in the first 1000 rows ;)
    uk_orgs = (
        get_crunchbase_orgs(nrows)
        .query("country == 'United Kingdom'")[["id", "name", "location_id"]]
        .copy()
    )
    # Load crunchbase location_id to NUTS2 mapping
    cb_nuts = get_crunchbase_nuts()
    # Add NUTS2 regions to company data
    uk_orgs = (
        uk_orgs.pipe(add_nuts_to_crunchbase, cb_nuts, 2010)
        .pipe(add_nuts_to_crunchbase, cb_nuts, 2013)
        .pipe(add_nuts_to_crunchbase, cb_nuts, 2016)
    )
    # Check how many organisations have been successfully mapped to NUTS2
    n_mapped = len(uk_orgs[uk_orgs.nuts2_2016 != -1])
    logging.info(
        f"{n_mapped}/{len(uk_orgs)} ({round(n_mapped/len(uk_orgs)*100)}%) organisations have been mapped to NUTS2"
    )
    # Load BEIS indicators
    indicators_df = get_beis_indicators()
    # Add BEIS indicators
    for indicator in indicators_df.indicator_id.unique():
        uk_orgs = add_indicator(uk_orgs, indicators_df, indicator, int(window_end_date))
    # Save the final table
    uk_orgs.to_csv(
        OUTPUT_FILEPATH / OUTPUT_FILENAME.format(window_end_date), index=False
    )


if __name__ == "__main__":
    typer.run(create_beis_indicators)
