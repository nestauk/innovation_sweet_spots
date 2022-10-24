from innovation_sweet_spots.tests.data_update_checks.utils import (
    run_filename_and_data_checks,
)
from innovation_sweet_spots import PROJECT_DIR
import typer

DATA_PATH = PROJECT_DIR / "inputs/data"


def run_update_checks(cur_dir: str, new_dir: str, dataset: str):
    """Run set of checks for current and new versions of a dataset.


    Args:
        cur_dir: Current version data directory
        new_dir: New version data directory
        dataset: "cb", "gtr" or "nihr"
    """
    cur_path = DATA_PATH / cur_dir
    new_path = DATA_PATH / new_dir
    run_filename_and_data_checks(cur_path, new_path)
    if dataset == "cb":
        pass
    if dataset == "gtr":
        pass
    if dataset == "nihr":
        pass


if __name__ == "__main__":
    typer.run(run_update_checks)
