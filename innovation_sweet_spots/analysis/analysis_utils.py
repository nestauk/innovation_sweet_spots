"""
Utils for doing data analysis

"""
from typing import Iterator
import pandas as pd
import numpy as np

import altair as alt

### Preparation


def create_documents_from_dataframe(
    df: pd.DataFrame, columns: Iterator[str]
) -> Iterator[str]:
    """Build documents from texts in the table columns"""
    # Select columns to include in the document
    df_ = df[columns].fillna("").copy()
    # Preprocess project text
    text_lists = [df_[col].to_list() for col in columns]
    # Create project documents
    docs = [preprocess_text(text) for text in create_documents(text_lists)]
    return docs


def create_documents(lists_of_texts: Iterator[str]) -> Iterator[str]:
    """
    Create documents from lists of texts for further analysis, e.g. to
    calculate tf-idf scores of n-grams. For example:
        (['one','two'], ['cat', 'dogs']) -> ['one cat', 'two dogs']

    Parameters
    ----------
        lists_of_texts:
            Contains lists of texts to be joined up and processed to create
            the "documents"; i-th element of each list corresponds to the
            i-th entity/document

    Yields
    ------
        Iterator[str]
            Document generator
    """
    # Check if all lists have the same length
    if len({len(i) for i in lists_of_texts}) == 1:
        # Transpose the lists of skill texts
        transposed_lists_of_texts = map(list, zip(*lists_of_texts))
        # Join up the skill texts for each skills entity
        return (
            " ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


def preprocess_text(text: str) -> str:
    """Placeholder for some more serious text preprocessing"""
    return text.lower().strip()


### Search term analysis


def is_term_present(search_term: str, docs: Iterator[str]) -> Iterator[bool]:
    """Simple method to check if a keyword or keyphrase is in the set of documents"""
    return [search_term in doc for doc in docs]


def search_via_docs(search_term: str, docs: Iterator[str], item_table: pd.DataFrame):
    """Returns table with only the items whose documents contain the search term"""
    bool_mask = is_term_present(search_term, docs)
    return item_table[bool_mask].copy()


def convert_date_to_year(str_date: str) -> int:
    """Convert string date of format YYYY-... to a year"""
    if type(str_date) is str:
        return int(str_date[0:4])
    else:
        return str_date


### Visualisations


def show_time_series(data, y, x="year"):
    chart = (
        alt.Chart(data, width=500).mark_line(point=True).encode(x=f"{x}:O", y=f"{y}:Q")
    )
    # return chart + chart.transform_loess(x, y, bandwidth=0.35).mark_line()
    return chart


### GTR specific utils


def link_gtr_projects_and_funds(
    gtr_funds: pd.DataFrame, link_gtr_funds: pd.DataFrame
) -> pd.DataFrame:
    """
    Links GTR project ids with their funding amount

    To do:
        - Inspect the reason for duplicate funds of the same amount for the same project
        - Check veracity of the approach (taking the highest income fund)
    """
    # Link project id and fund id
    gtr_project_to_fund = (
        gtr_funds.merge(link_gtr_funds)
        # Select the funds to include in the analysis
        .query("category=='INCOME_ACTUAL'")
        .sort_values("amount", ascending=False)
        .drop_duplicates("project_id", keep="first")
    )
    # Columns to use in downstream analysis
    fund_cols = ["project_id", "id", "rel", "category", "amount", "currencyCode"]
    return gtr_project_to_fund[fund_cols]


def get_gtr_project_funds(
    gtr_projects: pd.DataFrame, gtr_project_to_fund: pd.DataFrame
) -> pd.DataFrame:
    """Links GTR projects with their funding amount"""
    return (
        gtr_projects.merge(gtr_project_to_fund, on="project_id", how="left")
        .rename(columns={"rel": "rel_funds"})
        .drop("id", axis=1)
    )


def gtr_funding_per_year(
    funded_projects: pd.DataFrame, min_year: int = 2007, max_year: int = 2020
) -> pd.DataFrame:
    """
    Given a table with projects and their funds, return an aggregation by year
    """
    funded_projects["year"] = funded_projects.start.apply(convert_date_to_year)
    yearly_stats = (
        funded_projects.groupby("year")
        .agg(
            no_of_projects=("project_id", "count"),
            amount_total=("amount", "sum"),
            amount_median=("amount", np.median),
        )
        .reset_index()
        .query(f"year>={min_year}")
        .query(f"year<={max_year}")
    )
    # Add zero values for years without projects
    yearly_stats_imputed = (
        pd.DataFrame(data={"year": range(min_year, max_year + 1)})
        .merge(yearly_stats, how="left")
        .fillna(0)
        .astype({"no_of_projects": int, "amount_total": float, "amount_median": float})
    )
    return yearly_stats_imputed


def estimate_funding_level(
    yearly_stats: pd.DataFrame, past_years: int = 5, column: str = "amount_total"
):
    """Simple estimator of the funding level across past X years"""
    return yearly_stats.tail(past_years)[column].sum()


def estimate_growth_level(
    yearly_stats: pd.DataFrame, year_cycle: int = 5, column: str = "amount_total"
):
    """Compare two time periods (year cycles) and estimate growth"""
    first_cycle = yearly_stats.iloc[-year_cycle * 2 : -year_cycle][column].sum()
    second_cycle = yearly_stats.iloc[-year_cycle:][column].sum()
    growth = (second_cycle - first_cycle) / first_cycle
    return growth


def link_gtr_projects_and_orgs(
    gtr_organisations: pd.DataFrame, link_gtr_organisations: pd.DataFrame
) -> pd.DataFrame:
    """
    Links GTR project ids with their organisations
    """
    # Link project id and fund id
    gtr_project_to_org = link_gtr_organisations[["project_id", "id", "rel"]].merge(
        gtr_organisations[["id", "name"]], how="left"
    )
    # Columns to use in downstream analysis
    columns = ["project_id", "rel", "name"]
    return gtr_project_to_org[columns]


def get_gtr_project_orgs(
    gtr_projects: pd.DataFrame, project_to_org: pd.DataFrame
) -> pd.DataFrame:
    """Get organisations pertaining to a project"""
    projects_orgs = gtr_projects.merge(project_to_org, how="left").rename(
        columns={"rel": "rel_organisations"}
    )
    return projects_orgs


def get_org_stats(project_orgs_and_funds: pd.DataFrame) -> pd.DataFrame:
    """
    Characterise the organisations involved in the projects

    To do:
        - Inspect if organisation-level funding is also available

    """
    org_stats = (
        project_orgs_and_funds.groupby("name")
        .agg(no_of_projects=("project_id", "count"), amount_total=("amount", "sum"))
        .sort_values("no_of_projects", ascending=False)
    )
    return org_stats


### Hansard specific utils
"""
TODO:
PROJECTS
- Organisational network

HANSARD
- Break down by year, by person, by party
- Think about a baseline comparison

- Same sentiment analysis as with Guardian news

~~~~~~~~
Prelim analysis for detecting green documents
- Use my clustering algo with small kNN (perhaps = 5)
- Select clusters that are link to preselected Green topics
- TF_IDF vectorise and train the model

~~~~~~~~
Crunchbase > check keyword analysis, and get the same stats as projects
Add also VC names


"""
