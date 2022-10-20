from innovation_sweet_spots.analysis import analysis_utils as au
import pandas as pd


def get_time_series(
    research_project_funding,
    categories_to_check,
    taxonomy_level="Category",
    id_column="project_id",
):
    """Creates time series"""
    category_ts = []
    for tech_area in categories_to_check:
        # Select projects within a category
        df_category = research_project_funding.query(
            f"`{taxonomy_level}` == @tech_area"
        ).drop_duplicates([id_column, taxonomy_level])
        # Time series
        df_ts = (
            au.gtr_get_all_timeseries_period(
                df_category,
                period="year",
                min_year=2010,
                max_year=2022,
                start_date_column="start_date",
            )
            .assign(
                **{
                    taxonomy_level: tech_area,
                    "year": (lambda df: df.time_period.dt.year),
                }
            )
            .query("time_period <= '2021'")
        )
        category_ts.append(df_ts)
    category_ts = pd.concat(category_ts, ignore_index=False)
    return category_ts


def get_magnitude_vs_growth(
    category_ts, categories_to_check, taxonomy_level="Category", verbose=False
):
    """Creates magnitude vs growth plots"""
    category_magnitude_growth = []
    for tech_area in categories_to_check:
        if verbose:
            print(tech_area)
        df = (
            au.ts_magnitude_growth(
                category_ts.query(f"`{taxonomy_level}` == @tech_area").drop(
                    taxonomy_level, axis=1
                ),
                2017,
                2021,
            )
            .drop("index")
            .assign(**{taxonomy_level: tech_area})
        )
        category_magnitude_growth.append(df)
    return (
        pd.concat(category_magnitude_growth, ignore_index=False)
        .reset_index()
        .sort_values(["index", "magnitude"], ascending=False)
        # Convert to millions
        .assign(magnitude=lambda df: df.magnitude / 1000)
        # Convert to fractions
        .assign(growth=lambda df: df.growth / 100)
    )


def get_magnitude_vs_growth_plot(magnitude_growth, column="amount_total"):
    """Process the data for charts"""
    return magnitude_growth.query(f"index==@column")
