import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from innovation_sweet_spots import PROJECT_DIR, logger
from innovation_sweet_spots.tests.data_update_checks.utils import load_file_as_df
import pathlib

sns.set_theme(style="darkgrid")


def get_gtr_funding_per_year(path: pathlib.Path):
    """Group gtr funding by year and process into format
    that can be used in the function `gtr_funding_per_year_plot`"""
    return (
        load_file_as_df(path, "gtr_projects-wrangled_project_data.csv")[
            ["fund_start", "amount"]
        ]
        .assign(fund_start=lambda x: pd.to_datetime(x.fund_start, errors="coerce"))
        .dropna()
        .query("fund_start >= 20000101")
        .set_index("fund_start")
        .groupby(pd.Grouper(freq="Y"))["amount"]
        .sum()
        .reset_index()
        .assign(amount=lambda x: x.amount / 1_000_000_000)
        .rename(
            columns={
                "fund_start": "projects_started_in_year",
                "amount": "funding_amount_billions",
            }
        )
    )


def gtr_funding_per_year_plot(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Plot funding per year for new and current versions of the
    GtR dataset

    Args:
        cur_path: Path to the current version of the data
        new_path: Path to the new version of the data
    """
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        data=get_gtr_funding_per_year(cur_path),
        x="projects_started_in_year",
        y="funding_amount_billions",
        ax=ax[0],
    ).set(ylim=(0, None))
    ax[0].title.set_text(f"{cur_path.stem}/ funding per year")
    sns.lineplot(
        data=get_gtr_funding_per_year(new_path),
        x="projects_started_in_year",
        y="funding_amount_billions",
        ax=ax[1],
    ).set(ylim=(0, None))
    ax[1].title.set_text(f"{new_path.stem}/ funding per year")
    fig.set_size_inches(12, 5)
    fig.tight_layout()

    save_dir = (
        PROJECT_DIR / "innovation_sweet_spots/tests/data_update_checks/gtr/plots/"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_path = save_dir / "gtr_funding_per_year"
    fig.savefig(save_path)
    logger.info(f"GtR funding per year plots saved to {save_path}")


def get_gtr_projects_per_year(path: pathlib.Path):
    """Group gtr count of projects by year and process into format
    that can be used in the function `gtr_projects_per_year_plot`"""
    return (
        load_file_as_df(path, "gtr_projects-wrangled_project_data.csv")[
            ["fund_start", "project_id"]
        ]
        .assign(fund_start=lambda x: pd.to_datetime(x.fund_start, errors="coerce"))
        .dropna()
        .query("fund_start >= 20000101")
        .set_index("fund_start")
        .groupby(pd.Grouper(freq="Y"))["project_id"]
        .count()
        .reset_index()
        .rename(
            columns={
                "fund_start": "year",
                "project_id": "n_projects_started",
            }
        )
    )


def gtr_projects_per_year_plot(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Plot projects per year for new and current versions of the
    GtR dataset

    Args:
        cur_path: Path to the current version of the data
        new_path: Path to the new version of the data
    """
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        data=get_gtr_projects_per_year(cur_path),
        x="year",
        y="n_projects_started",
        ax=ax[0],
    ).set(ylim=(0, None))
    ax[0].title.set_text(f"{cur_path.stem}/ projects started per year")
    sns.lineplot(
        data=get_gtr_projects_per_year(new_path),
        x="year",
        y="n_projects_started",
        ax=ax[1],
    ).set(ylim=(0, None))
    ax[1].title.set_text(f"{new_path.stem}/ projects started per year")
    fig.set_size_inches(12, 5)
    fig.tight_layout()

    save_dir = (
        PROJECT_DIR / "innovation_sweet_spots/tests/data_update_checks/gtr/plots/"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_path = save_dir / "gtr_projects_per_year"
    fig.savefig(save_path)
    logger.info(f"GtR projects per year plots saved to {save_path}")
