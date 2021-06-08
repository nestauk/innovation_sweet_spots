# Generic scripts to get DAPS tables
import logging
import os
from configparser import ConfigParser
from typing import Any, Dict, Iterator

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

# from pandas._typing import FilePathOrBuffer  # Not available in pandas < 1

MYSQL_CONFIG = os.environ["MYSQL_CONFIG"]


def get_engine(config_path, database="production", **engine_kwargs):
    """Get a SQL alchemy engine from config"""
    cp = ConfigParser()
    cp.read(config_path)
    cp = cp["client"]
    url = URL(
        drivername="mysql+pymysql",
        database=database,
        username=cp["user"],
        host=cp["host"],
        password=cp["password"],
    )
    return create_engine(url, **engine_kwargs)


def fetch_daps_table(table_name: str, fields: str = "all") -> pd.DataFrame:
    """Fetch DAPS tables if we don't have them already
    Args:
        table_name: name
        path: path for the table
        fields: fields to fetch. If a list, fetches those
    Returns:
        table
    """
    logging.info(f"Fetching {table_name}")
    engine = get_engine(MYSQL_CONFIG)
    con = engine.connect().execution_options(stream_results=True)

    if fields == "all":
        chunks = pd.read_sql_table(table_name, con, chunksize=1000)
    else:
        chunks = pd.read_sql_table(table_name, con, columns=fields, chunksize=1000)

    return chunks


def stream_df_to_csv(
    df_iterator: Iterator[pd.DataFrame],
    path_or_buf: Any,  # FilePathOrBuffer
    **kwargs,
):
    """Stream a DataFrame iterator to csv.

    Args:
        df_iterator: DataFrame chunks to stream to CSV
        path_or_buf: FilePath or Buffer (passed to `DataFrame.to_csv`)
        kwargs: Extra args passed to `DataFrame.to_csv`. Cannot contain
            any of `{"mode", "header", "path_or_buf"}` - `mode` is "a" and
            `header` is `False` for all but initial chunks.


    Raises:
        ValueError if `kwargs` contains disallowed values.
    """
    if any((key in kwargs for key in ["mode", "header", "path_or_buf"])):
        raise ValueError()

    # First chunk: mode "w" and write column names
    initial = next(df_iterator)
    initial.to_csv(path_or_buf, **kwargs)
    # Subsequent chunks:
    for chunk in df_iterator:
        chunk.to_csv(path_or_buf, mode="a", header=False, **kwargs)


def save_daps_table(table: pd.DataFrame, name: str, path: str):
    """Save DAPS tables
    Args:
        table: table to save
        name: table name
        path: directory where we store the table
    """
    table.to_csv(f"{path}/{name}.csv", index=False)
