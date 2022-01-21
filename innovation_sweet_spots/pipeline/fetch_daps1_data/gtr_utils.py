# Fetch GtR tables
from pathlib import Path
from typing import Dict, Iterator

import pandas as pd

from daps1_utils import (
    get_engine,
    fetch_daps_table,
    MYSQL_CONFIG,
    stream_df_to_csv,
)

GTR_PATH = Path(__file__).parents[3] / "inputs/data/gtr"

GTR_TABLES = [
    "gtr_funds",
    "gtr_topic",
    "gtr_organisations",
    "gtr_organisations_locations",
    "gtr_participant",
    "gtr_persons",
    "gtr_link_table",
]  # NB: Projects table is processed differently


def projects_funded_from_2006() -> Iterator[pd.DataFrame]:
    """GtR projects with funding starting from 2006.

    Returns:
        Iterable of query results
    """
    engine = get_engine(MYSQL_CONFIG)
    con = engine.connect().execution_options(stream_results=True)
    query = """
    SELECT
        DISTINCT gtr_projects.id AS project_id,
        gtr_projects.title,
        gtr_projects.grantCategory,
        gtr_projects.leadFunder,
        gtr_projects.abstractText,
        gtr_projects.potentialImpact,
        gtr_projects.techAbstractText,
        gtr_funds.start
    FROM
        gtr_projects
            INNER JOIN
                (SELECT * FROM gtr_link_table
                 WHERE gtr_link_table.table_name = 'gtr_funds')
                AS l
                ON l.project_id = gtr_projects.id
            INNER JOIN
                (SELECT * FROM gtr_funds
                 WHERE YEAR(gtr_funds.start) > 2006)
                AS gtr_funds
                ON l.id = gtr_funds.id
    GROUP BY gtr_projects.id HAVING MIN(YEAR(gtr_funds.start));
    """

    return pd.read_sql_query(query, con, chunksize=1000)


def fetch_save_table(name: str):
    link = fetch_daps_table(name)
    stream_df_to_csv(link, f"{GTR_PATH}/{name}.csv", index=False)


def fetch_save_gtr_tables():
    for table in GTR_TABLES:
        fetch_save_table(table)
    # NB: Projects table is processed differently
    projects_filtered = projects_funded_from_2006()
    stream_df_to_csv(projects_filtered, f"{GTR_PATH}/gtr_projects.csv", index=False)


def get_names(con) -> Dict[str, str]:
    """Fetch non-null `{id: name}` pairs from gtr_organisations."""

    return (
        pd.read_sql_table("gtr_organisations", con, columns=["id", "name"])
        .set_index("id")
        .name.dropna()
        .to_dict()
    )
