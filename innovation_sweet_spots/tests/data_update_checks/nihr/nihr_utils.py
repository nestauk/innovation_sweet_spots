import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from innovation_sweet_spots import PROJECT_DIR, logger
from innovation_sweet_spots.tests.data_update_checks.utils import load_file_as_df
import pathlib

sns.set_theme(style="darkgrid")


def get_nihr_funding_per_year(path: pathlib.Path):
    """Group NIHR funding by year and process into format
    that can be used in the function `nihr_awarded_amount_per_year_plot`"""
    return (
        load_file_as_df(path, "nihr_summary_data.csv")[["start_date", "award_amount_m"]]
        .assign(start_date=lambda x: pd.to_datetime(x.start_date, errors="coerce"))
        .dropna()
        .query("start_date >= 20000101")
        .set_index("start_date")
        .groupby(pd.Grouper(freq="Y"))["award_amount_m"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "start_date": "projects_started_in_year",
                "award_amount_m": "award_amount_millions",
            }
        )
    )


def nihr_awarded_amount_per_year_plot(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Plot awarded amount per year for new and current versions of the
    NIHR dataset

    Args:
        cur_path: Path to the current version of the data
        new_path: Path to the new version of the data
    """
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        data=get_nihr_funding_per_year(cur_path),
        x="projects_started_in_year",
        y="award_amount_millions",
        ax=ax[0],
    ).set(ylim=(0, None))
    ax[0].title.set_text(f"{cur_path.stem}/ amounts awarded per year")
    sns.lineplot(
        data=get_nihr_funding_per_year(new_path),
        x="projects_started_in_year",
        y="award_amount_millions",
        ax=ax[1],
    ).set(ylim=(0, None))
    ax[1].title.set_text(f"{new_path.stem}/ amounts awarded per year")
    fig.set_size_inches(12, 5)
    fig.tight_layout()

    save_dir = (
        PROJECT_DIR / "innovation_sweet_spots/tests/data_update_checks/nihr/plots/"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_path = save_dir / "nihr_awarded_amount_per_year"
    fig.savefig(save_path)
    logger.info(f"NIHR amounts awarded per year plots saved to {save_path}")


def get_nihr_projects_per_year(path: pathlib.Path):
    """Group NIHR count of projects by year and process into format
    that can be used in the function `nihr_projects_per_year_plot`"""
    return (
        load_file_as_df(path, "nihr_summary_data.csv")[["start_date", "project_id"]]
        .assign(start_date=lambda x: pd.to_datetime(x.start_date, errors="coerce"))
        .dropna()
        .query("start_date >= 20000101")
        .set_index("start_date")
        .groupby(pd.Grouper(freq="Y"))["project_id"]
        .count()
        .reset_index()
        .rename(
            columns={
                "start_date": "year",
                "project_id": "n_projects_started",
            }
        )
    )


def nihr_projects_per_year_plot(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Plot number of projects per year for new and current versions of the
    NIHR dataset

    Args:
        cur_path: Path to the current version of the data
        new_path: Path to the new version of the data
    """
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        data=get_nihr_projects_per_year(cur_path),
        x="year",
        y="n_projects_started",
        ax=ax[0],
    ).set(ylim=(0, None))
    ax[0].title.set_text(f"{cur_path.stem}/ projects started per year")
    sns.lineplot(
        data=get_nihr_projects_per_year(new_path),
        x="year",
        y="n_projects_started",
        ax=ax[1],
    ).set(ylim=(0, None))
    ax[1].title.set_text(f"{new_path.stem}/ projects started per year")
    fig.set_size_inches(12, 5)
    fig.tight_layout()

    save_dir = (
        PROJECT_DIR / "innovation_sweet_spots/tests/data_update_checks/nihr/plots/"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_path = save_dir / "nihr_projects_started_per_year"
    fig.savefig(save_path)
    logger.info(f"NIHR projects started per year plots saved to {save_path}")
