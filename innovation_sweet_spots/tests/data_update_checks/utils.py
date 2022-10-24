from innovation_sweet_spots import PROJECT_DIR, logger
import pathlib
import pandas as pd
import seaborn as sns
from innovation_sweet_spots.utils.io import load_json

sns.set_theme(style="whitegrid")

PLOTS_DIR = PROJECT_DIR / ("innovation_sweet_spots/tests/data_update_checks/plots")


def load_file_as_df(dir_path: pathlib.Path, filename: str) -> pd.DataFrame:
    """Load file as dataframe"""
    path = dir_path / filename
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.DataFrame(load_json(path))


def find_filenames(
    path: pathlib.Path,
) -> list:
    """Return a set of filenames from path"""
    return {file.name for file in path.iterdir()}


def check_filenames_match(cur_files: set, new_files: set):
    """Check that current and new sets of filenames are the same.
    Log the outcome and provide additional information on the
    differences."""
    logger.info("Checking filenames match...")
    if cur_files == new_files:
        logger.info("Current and new files match.")
    else:
        logger.warning("Current and new files do not match.")
        if missing_files := cur_files - new_files:
            logger.warning(
                f"The new dataset is missing {len(missing_files)} files: {missing_files}."
            )
        if add_files := new_files - cur_files:
            logger.warning(
                f"The new dataset has {len(add_files)} additional files: {add_files}."
            )


def check_n_rows(cur_df: pd.DataFrame, new_df: pd.DataFrame, filename: str):
    """Check that the number of rows in the current and new dataframes are
    the same. Log the outcome and provide additional information on the
    differences."""
    additional_rows = len(new_df) - len(cur_df)
    start_message = f"The new version of the file {filename} has"
    end_message = "rows of data compared to the previous version"
    if additional_rows > 0:
        logger.info(f"{start_message} {additional_rows} additional {end_message}.")
    elif additional_rows < 0:
        logger.warning(f"{start_message} {additional_rows} fewer {end_message}.")
    else:
        logger.warning(f"{start_message} the same number of {end_message}.")


def check_cols_match(cur_df: pd.DataFrame, new_df: pd.DataFrame, filename: str):
    """Check that the number of columns in the current and new dataframes are
    the ame. Log the outcome and provide additional information on the
    differences."""
    cur_cols = set(cur_df.columns)
    new_cols = set(new_df.columns)
    if cur_cols == new_cols:
        logger.info(f"{filename} current and new columns match.")
    else:
        logger.warning(f"{filename} current and new columns do not match.")
        if missing_cols := cur_cols - new_cols:
            logger.warning(
                f"{filename} new dataset is missing {len(missing_cols)} columns: {missing_cols}."
            )
        if add_cols := new_cols - cur_cols:
            logger.warning(
                f"{filename} new dataset has {len(add_cols)} additional columns: {add_cols}."
            )


def calc_percent_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Return percentage of nans in each column for provided dataframe."""
    return (
        (df.isnull().sum() / len(df) * 100)
        .reset_index()
        .rename(columns={"index": "column", 0: "% nans"})
    )


def combine_data_for_nan_plot(
    cur_df: pd.DataFrame, new_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine new and current dataframes and add a version column."""
    return pd.concat(
        [
            calc_percent_nans(new_df).assign(version="new_data"),
            calc_percent_nans(cur_df).assign(version="current_data"),
        ]
    )


def nan_plot(combined_data: pd.DataFrame, filename: str):
    """Plot nan percentage for each column, comparing the new
    and current data"""
    nan_plot = sns.catplot(
        data=combined_data,
        kind="bar",
        x="% nans",
        y="column",
        hue="version",
        orient="h",
        height=7,
        palette="Paired",
    ).set(
        title=f"Percentage of NaNs in current vs. new data -- {filename}", xlim=(0, 1)
    )
    save_lbl = filename.split(".")[0]
    save_path = PLOTS_DIR / f"{save_lbl}_nan_plot.png"
    nan_plot.figure.savefig(save_path, bbox_inches="tight")
    logger.info(f"NaN plot saved to {save_path}")


def run_data_checks(
    cur_path: pathlib.Path, new_path: pathlib.Path, matching_files: set
):
    """Run checks comparing new and current datasets for number of rows,
    missing/additional columns and save nan plots for each column.

    Args:
        cur_path: Path to the current version of the dataset
        new_path: Path to the new version of the dataset
        matching_files: Set of shared files between the current
            and new datasets
    """
    for file in matching_files:
        logger.info(f"Checking file {file}...")
        cur_df = load_file_as_df(cur_path, file)
        new_df = load_file_as_df(new_path, file)

        check_n_rows(cur_df, new_df, file)

        #### Remove this line... use for testing functions
        # new_df["extra_col"] = 100
        # cur_df["missing_col"] = 50

        check_cols_match(cur_df, new_df, file)

        nan_data = combine_data_for_nan_plot(cur_df, new_df)
        nan_plot(nan_data, file)


def run_filename_and_data_checks(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Check that current and new sets of filenames are the same.
    Check new and current datasets for number of rows,
    missing/additional columns and save nan plots for each column.s

    Args:
        cur_path: Path to the current version of the dataset
        new_path: Path to the new version of the dataset
    """
    cur_files = find_filenames(cur_path)
    new_files = find_filenames(new_path)
    check_filenames_match(cur_files, new_files)
    matching_files = cur_files.intersection(new_files)
    run_data_checks(cur_path, new_path, matching_files)
