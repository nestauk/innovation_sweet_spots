# Dataset Update Checks

`data_update_checks.py` compares the current version of a dataset with a new version. It checks for missing/additional files, rows, columns, NaNs. It also creates some dataset specific plots to see how the coverage for each year has changed.

## Requirements

In addition to the main Innovation Sweet Spots requirements, there are some additional requirements needed to run the dataset update checks. To install these, run: `pip install -r innovation_sweet_spots/tests/data_update_checks/requirements.txt`.

## Usage

Run: `python innovation_sweet_spots/tests/data_update_checks/data_update_checks.py [current_directory] [new_directory] [dataset]`

`[current_directory]` and `[new_directory]` are assumed to be in `inputs/data`.

`[dataset]` can be either `cb`, `gtr` or `nihr`.

For example: `python innovation_sweet_spots/tests/data_update_checks/data_update_checks.py cb_2021 cb_2022_june cb`
