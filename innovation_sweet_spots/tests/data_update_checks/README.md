# Dataset Update Checks

`data_update_checks.py` compares two versions of a dataset (e.g. an old version with a new version). It checks for missing/additional files, rows, columns, NaNs. It also creates some dataset specific plots to see how the coverage for each year has changed.

## Usage

Run: `python innovation_sweet_spots/tests/data_update_checks/data_update_checks.py [old_version_directory] [new_version_directory] [dataset]`

`[old_version_directory]` and `[new_version_directory]` are assumed to be in `inputs/data`.

`[dataset]` can be either `cb`, `gtr` or `nihr`.

For example: `python innovation_sweet_spots/tests/data_update_checks/data_update_checks.py cb_2021 cb_2022_june cb`
