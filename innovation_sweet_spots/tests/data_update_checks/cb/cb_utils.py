import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from innovation_sweet_spots import PROJECT_DIR, logger
from innovation_sweet_spots.tests.data_update_checks.utils import load_file_as_df
import pathlib

sns.set_theme(style="darkgrid")


def get_cb_raised_per_year(path: pathlib.Path):
    """Group funding raised amount by year and process into format
    that can be used in the function `cb_raised_per_year_plot`"""
    return (
        load_file_as_df(path, "crunchbase_funding_rounds.csv")[
            ["announced_on", "raised_amount_usd"]
        ]
        .assign(announced_on=lambda x: pd.to_datetime(x.announced_on, errors="coerce"))
        .dropna()
        .query("announced_on >= 20000101")
        .set_index("announced_on")
        .groupby(pd.Grouper(freq="Y"))["raised_amount_usd"]
        .sum()
        .reset_index()
        .assign(raised_amount_usd=lambda x: x.raised_amount_usd / 1_000_000_000)
        .rename(
            columns={
                "announced_on": "year",
                "raised_amount_usd": "raised_amount_usd_billions",
            }
        )
    )


def cb_raised_per_year_plot(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Plot funding raised per year for new and current versions of the
    Crunchbase dataset

    Args:
        cur_path: Path to the current version of the data
        new_path: Path to the new version of the data
    """
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        data=get_cb_raised_per_year(cur_path),
        x="year",
        y="raised_amount_usd_billions",
        ax=ax[0],
    ).set(ylim=(0, None))
    ax[0].title.set_text(f"{cur_path.stem}/ funding raised per year")
    sns.lineplot(
        data=get_cb_raised_per_year(new_path),
        x="year",
        y="raised_amount_usd_billions",
        ax=ax[1],
    ).set(ylim=(0, None))
    ax[1].title.set_text(f"{new_path.stem}/ funding raised per year")
    fig.set_size_inches(9, 4)
    fig.tight_layout()
    save_dir = (
        PROJECT_DIR
        / f"innovation_sweet_spots/tests/data_update_checks/cb/{cur_path.stem}_vs_{new_path.stem}_plots/"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_path = save_dir / "crunchbase_funding_raised_per_year"
    fig.savefig(save_path)
    logger.info(f"Crunchbase funding raised per year plots saved to {save_path}")


def get_cb_companies_founded(path: pathlib.Path):
    """Group number of companies founded by year and process into format
    that can be used in the function `cb_companies_per_year_plot`"""
    return (
        load_file_as_df(path, "crunchbase_orgs.csv")[["founded_on", "id"]]
        .assign(founded_on=lambda x: pd.to_datetime(x.founded_on, errors="coerce"))
        .dropna()
        .query("founded_on >= 20000101")
        .set_index("founded_on")
        .groupby(pd.Grouper(freq="Y"))["id"]
        .count()
        .reset_index()
        .rename(columns={"founded_on": "year", "id": "n_companies_founded"})
    )


def cb_companies_per_year_plot(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Plot number of companies founded per year for the new and current versions
    of the Crunchbase dataset

    Args:
        cur_path: Path to the current version of the data
        new_path: Path to the new version of the data
    """
    fig, ax = plt.subplots(1, 2)
    sns.lineplot(
        data=get_cb_companies_founded(cur_path),
        x="year",
        y="n_companies_founded",
        ax=ax[0],
    ).set(ylim=(0, None))
    ax[0].title.set_text(f"{cur_path.stem}/ number of companies founded per year")
    sns.lineplot(
        data=get_cb_companies_founded(new_path),
        x="year",
        y="n_companies_founded",
        ax=ax[1],
    ).set(ylim=(0, None))
    ax[1].title.set_text(f"{new_path.stem}/ number of companies founded per year")
    fig.set_size_inches(9, 4)
    fig.tight_layout()
    save_dir = (
        PROJECT_DIR
        / f"innovation_sweet_spots/tests/data_update_checks/cb/{cur_path.stem}_vs_{new_path.stem}_plots/"
    )
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    save_path = save_dir / "crunchbase_n_companies_founded_per_year"
    fig.savefig(save_path)
    logger.info(
        f"Crunchbase number of companies founded per year plots saved to {save_path}"
    )


def cb_plots(cur_path: pathlib.Path, new_path: pathlib.Path):
    """Run `cb_companies_per_year_plot` and `cb_raised_per_year_plot` functions
    and except FileNotFoundErrors"""
    try:
        cb_raised_per_year_plot(cur_path, new_path)
    except FileNotFoundError as e:
        logger.warning(f"{e}. Cannot make Crunchbase raised per year plot.")

    try:
        cb_companies_per_year_plot(cur_path, new_path)
    except FileNotFoundError as e:
        logger.warning(f"{e}. Cannot make Crunchbase companies founded per year plot.")
