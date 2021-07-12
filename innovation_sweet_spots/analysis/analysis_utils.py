"""
Utils for doing data analysis

"""
from innovation_sweet_spots import logging
from innovation_sweet_spots.utils.text_cleaning_utils import clean_text

from typing import Iterator
import pandas as pd
import numpy as np
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import altair as alt

### Preparation


def create_documents_from_dataframe(
    df: pd.DataFrame, columns: Iterator[str], preprocessor=None
) -> Iterator[str]:
    """Build documents from texts in the table columns"""
    # Select columns to include in the document
    df_ = df[columns].fillna("").copy()
    # Preprocess project text
    text_lists = [df_[col].to_list() for col in columns]
    # Create project documents
    docs = [preprocessor(text) for text in create_documents(text_lists)]
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
            ". ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


# Text preprocessors
def preprocess_text(text: str) -> str:
    """Placeholder for some more serious text preprocessing"""
    return text.lower().strip()


def preprocess_text_clean(text: str) -> str:
    return clean_text(text)


def preprocess_text_clean_sentences(text: str) -> Iterator[str]:
    # Split text into sentences
    sents = (text for text in split_sentences(text) if len(text.split(" ")) > 2)
    # Clean each sentence
    clean_sents = [clean_text(sent) for sent in sents]
    return clean_sents


### Search term analysis


def is_term_present(search_term: str, docs: Iterator[str]) -> Iterator[bool]:
    """Simple method to check if a keyword or keyphrase is in the set of documents"""
    return [search_term in doc for doc in docs]


def is_term_present_in_sentences(
    search_term: str, docs: Iterator[Iterator[str]], min_mentions: int = 1
) -> Iterator[bool]:
    """Simple method to check if a keyword or keyphrase is in the set of documents"""
    is_term_in_doc = (is_term_present(search_term, sentences) for sentences in docs)
    return [sum(s) >= min_mentions for s in is_term_in_doc]


def get_docs_with_term(search_term: str, docs: Iterator[str]):
    is_term = is_term_present(search_term, docs)
    return [doc for i, doc in enumerate(docs) if is_term[i]]


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


def impute_years(
    yearly_stats: pd.DataFrame, min_year: int, max_year: int, trim_bounds: bool = False
):
    """Add zero values for years without data"""
    if trim_bounds:
        min_year = max(yearly_stats.year.min(), min_year)
        max_year = min(yearly_stats.year.max(), max_year)
    return (
        pd.DataFrame(data={"year": range(min_year, max_year + 1)})
        .merge(yearly_stats, how="left")
        .fillna(0)
    )


def split_sentences(doc):
    return [sent.strip() for sent in re.split("\.|\?|\!|\;|\•", doc)]


def get_sentences_with_term(search_term, docs):
    """ """
    sentences_with_term = []
    for doc in docs:
        for sent in split_sentences(doc):
            if search_term in sent:
                sentences_with_term.append(sent)
    return sentences_with_term


# def get_span_with_term(search_term, docs):
#     """ """
#     spans_with_term = []
#     for doc in docs:
#         for sent in split_sentences(doc):
#             if search_term in sent:
#                 sentences_with_term.append(sent)
#     return sentences_with_term


def get_document_sentences_with_term(search_term, docs):
    """Test out!"""
    sentences_with_term = []
    for doc in docs:
        doc_sentences = []
        for sent in split_sentences(doc):
            if search_term in sent:
                doc_sentences.append(sent)
        sentences_with_term.append(doc_sentences)
    return sentences_with_term


def get_sentence_sentiment(sentences: Iterator[str]) -> pd.DataFrame:
    """Calculates sentiment for each sentence in the list, and sorts them"""
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(sentence) for sentence in sentences]
    # Optional: output as a dataframe
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_df["sentences"] = sentences
    sentiment_df = sentiment_df.sort_values("compound")
    return sentiment_df


def cleanhtml(raw_html: str) -> str:
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext


### Visualisations


def capitalise_label(text, units=None):
    new_text = " ".join(text.split("_")).capitalize()
    if units is not None:
        new_text += f" ({units})"
    return new_text


def show_time_series(data, y, x="year"):
    chart = (
        alt.Chart(data, width=500, height=200)
        .mark_line(point=True)
        .encode(x=f"{x}:O", y=f"{y}:Q")
    )
    # return chart + chart.transform_loess(x, y, bandwidth=0.35).mark_line()
    return chart


def show_time_series_fancier(data, y, x="year", show_trend=True):
    chart = (
        alt.Chart(data, width=400, height=200)
        .mark_line(point=False, stroke="#3e0c59", strokeWidth=1.5)
        .encode(alt.X(f"{x}:O"), alt.Y(f"{y}:Q"))
    )
    chart.encoding.x.title = capitalise_label(x)
    capitalise_label
    if "amount" in y:
        chart.encoding.y.title = capitalise_label(y, units="thousands")
    else:
        chart.encoding.y.title = capitalise_label(y)
    if show_trend:
        chart_trend = chart.transform_regression(x, y).mark_line(
            stroke="black", strokeDash=[2, 2], strokeWidth=0.5
        )
        chart = alt.layer(chart, chart_trend)

    return chart


def show_time_series_points(data, y, x="year", ymax=None, clip=True):
    if ymax is None:
        ymax = data[y].max()
    base = (
        alt.Chart(data, width=400, height=200)
        .mark_point(opacity=0.8, size=20, clip=clip, color="#6295c4")
        .encode(
            alt.X("year:O"),
            alt.Y("amount:Q", scale=alt.Scale(domain=(0, ymax))),
            tooltip=["title"],
        )
    )
    base.encoding.x.title = capitalise_label(x)
    base.encoding.y.title = capitalise_label(y, "thousands")
    base
    composite = alt.layer(
        base,
        base.transform_loess("year", "amount").mark_line(color="#3e0c59", size=1.5),
    )
    fig = nicer_axis(composite)
    return fig


def nicer_axis(fig):
    return fig.configure_axis(
        labelFontSize=13,
        titleFontSize=13,
        labelFontWeight="lighter",
        titleFontWeight="normal",
    )


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
    yearly_stats.amount_total = yearly_stats.amount_total / 1000
    yearly_stats.amount_median = yearly_stats.amount_median / 1000
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
    """Compare two time periods (year cycles) and estimate growth percentage"""
    first_cycle = yearly_stats.iloc[-year_cycle * 2 : -year_cycle][column].sum()
    second_cycle = yearly_stats.iloc[-year_cycle:][column].sum()
    growth = (second_cycle - first_cycle) / first_cycle * 100
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


def link_gtr_projects_and_topics(gtr_projects, gtr_topics, link_gtr_topics):
    gtr_project_topics = (
        gtr_projects.merge(link_gtr_topics, how="left")
        .merge(gtr_topics, how="left")
        .drop(["id", "table_name", "rel"], axis=1)
    )
    return gtr_project_topics


### Crunchbase specific utils


def fill_na_funds(orgs):
    for col in ["raised_amount", "total_funding"]:
        if col in orgs.columns:
            orgs[col] = orgs[col].fillna(0).astype(float)
            orgs[col + "_usd"] = orgs[col + "_usd"].fillna(0).astype(float)
    return orgs


def cb_orgs_with_most_funding(orgs):
    columns = [
        "name",
        "city",
        "founded_on",
        "num_funding_rounds",
        "total_funding",
        "total_funding_currency_code",
        "total_funding_usd",
    ]
    orgs = (
        fill_na_funds(orgs.copy())
        .sort_values("total_funding_usd", ascending=False)
        .reset_index(drop=True)
    )
    return orgs[columns]


def cb_orgs_funded_by_year(
    orgs: pd.DataFrame, min_year: int = 2007, max_year: int = 2020
):
    orgs = orgs[
        -orgs.founded_on.isnull()
    ].copy()  # Some orgs don't have year when they were founded...
    orgs["year"] = orgs.founded_on.apply(convert_date_to_year).astype(int)
    yearly_founded_orgs = (
        orgs.groupby("year").agg(no_of_orgs_founded=("id", "count")).reset_index()
    )
    yearly_founded_orgs = impute_years(yearly_founded_orgs, min_year, max_year)
    return yearly_founded_orgs


def get_cb_org_funding_rounds(
    orgs: pd.DataFrame, cb_funding_rounds: pd.DataFrame
) -> pd.DataFrame:
    """Add funding round information to crunchbase organisations"""
    fund_rounds = (
        orgs[["id", "name"]]
        .rename(columns={"id": "org_id"})
        .merge(
            cb_funding_rounds[
                [
                    "id",
                    "org_id",
                    "announced_on",
                    "investment_type",
                    "raised_amount",
                    "raised_amount_currency_code",
                    "raised_amount_usd",
                ]
            ],
            on="org_id",
        )
        .rename(columns={"id": "funding_round_id"})
        .sort_values("announced_on")
    )
    return fund_rounds


def check_currencies(fund_rounds):
    """# Check if there are different currencies present"""
    currencies = fund_rounds[
        -fund_rounds.raised_amount_currency_code.isnull()
    ].raised_amount_currency_code.unique()
    if len(currencies) > 1:
        logging.warning(f"More than one unique currency: {list(currencies)}")


def get_cb_funding_per_year(
    fund_rounds: pd.DataFrame, min_year: int = 2007, max_year: int = 2020
) -> pd.DataFrame:
    """Calculate raised amount of money across all orgs"""
    fund_rounds["year"] = fund_rounds.announced_on.apply(convert_date_to_year)

    check_currencies(fund_rounds)

    yearly_stats = (
        fund_rounds.groupby("year")
        .agg(
            no_of_rounds=("funding_round_id", "count"),
            raised_amount_total=("raised_amount", "sum"),
            raised_amount_usd_total=("raised_amount_usd", "sum"),
        )
        .reset_index()
        .query(f"year>={min_year}")
        .query(f"year<={max_year}")
    )

    yearly_stats_imputed = impute_years(
        yearly_stats, min_year, max_year, trim_bounds=False
    ).astype(
        {
            "no_of_rounds": int,
            "raised_amount_total": float,
            "raised_amount_usd_total": float,
        }
    )
    return yearly_stats_imputed


def get_funding_round_investors(
    fund_rounds: pd.DataFrame, cb_investments: pd.DataFrame
) -> pd.DataFrame:
    """ """
    fund_rounds_investors = fund_rounds.merge(
        cb_investments[
            ["funding_round_id", "investor_name", "id", "investor_type", "partner_name"]
        ],
        on="funding_round_id",
    )
    fund_rounds_investors.raised_amount = fund_rounds_investors.raised_amount.fillna(0)
    fund_rounds_investors.raised_amount = (
        fund_rounds_investors.raised_amount_usd.fillna(0)
    )
    return fund_rounds_investors


def investor_raised_amounts(
    fund_rounds_investors: pd.DataFrame,
) -> pd.DataFrame:
    """NB: Raised amounts are inflated at the moment, showing the rounds"""
    check_currencies(fund_rounds_investors)
    investors = (
        fund_rounds_investors.groupby(["investor_name"])
        .agg(
            no_of_rounds=("funding_round_id", "count"),
            total_round_value=("raised_amount", "sum"),
            total_round_value_usd=("raised_amount_usd", "sum"),
        )
        .reset_index()
        .sort_values("total_round_value_usd", ascending=False)
    )
    return investors


### Hansard specific utils


def get_hansard_mentions_per_year(
    speeches: pd.DataFrame, min_year: int = 2007, max_year: int = 2020
) -> pd.DataFrame:
    """ """
    min_year = max(speeches.year.min(), min_year)
    max_year = min(speeches.year.max(), max_year)

    yearly_mentions = (
        speeches.groupby("year").agg(mentions=("id", "count")).reset_index()
    )
    # Add zero values for years without projects
    yearly_mentions_imputed = (
        pd.DataFrame(data={"year": range(min_year, max_year + 1)})
        .merge(yearly_mentions, how="left")
        .fillna(0)
        .astype({"mentions": int})
    )
    return yearly_mentions_imputed


def get_hansard_mentions_per_party(speeches):
    mentions = (
        speeches.groupby("party")
        .agg(mentions=("id", "count"))
        .reset_index()
        .sort_values("party")
    )
    return mentions


def get_hansard_mentions_per_person(speeches):
    mentions = (
        speeches.groupby("speakername")
        .agg(counts=("id", "count"))
        .reset_index()
        .sort_values("speakername")
    )
    return mentions


### Guardian specific utils


def get_guardian_mentions_per_year(
    articles: Iterator[dict], min_year: int = 2007, max_year: int = 2020
) -> pd.DataFrame:
    """ """
    dates = [
        convert_date_to_year(article["webPublicationDate"]) for article in articles
    ]
    counts = dict(Counter(dates))
    yearly_mentions = pd.DataFrame(
        data={"year": counts.keys(), "articles": counts.values()}
    )
    # Add zero values for years without mentions
    min_year = max(yearly_mentions.year.min(), min_year)
    max_year = min(yearly_mentions.year.max(), max_year)
    yearly_mentions_imputed = (
        pd.DataFrame(data={"year": range(min_year, max_year + 1)})
        .merge(yearly_mentions, how="left")
        .fillna(0)
        .astype({"articles": int})
    )
    return yearly_mentions_imputed


def get_guardian_contributors(articles):
    tags = (a["tags"] for a in articles)
    contributors = (tag[0]["webTitle"] for tag in tags if len(tag) > 0)
    counts = dict(Counter(contributors))
    contributor_table = pd.DataFrame(
        data={"contributor": counts.keys(), "no_of_articles": counts.values()}
    ).sort_values("no_of_articles", ascending=False)
    return contributor_table


def get_guardian_sentences_with_term(search_term, articles, field="body"):
    article_texts = get_article_field(articles, field=field)
    sentences = get_sentences_with_term(search_term, article_texts)
    return sentences


def get_article_field(articles, field="headline"):
    article_texts = (cleanhtml(preprocess_text(a["fields"][field])) for a in articles)
    return article_texts


def news_sentiment_over_years(search_term, articles):
    article_docs = list(get_article_field(articles, field="body"))
    doc_sents = get_document_sentences_with_term(search_term, article_docs)
    doc_years = [convert_date_to_year(a["webPublicationDate"]) for a in articles]
    # more_than_once = np.array([len(d) for d in doc_sents])>1
    df = pd.DataFrame(
        [(doc_years[i], sent) for i, doc in enumerate(doc_sents) for sent in doc],
        columns=["year", "sentences"],
    )
    df = df.merge(get_sentence_sentiment(df.sentences.to_list()))
    df = df.groupby("year").agg(mean_sentiment=("compound", "mean")).reset_index()
    return df


def articles_table(articles: Iterator[dict]):
    """Creates a dataframe with article headlines and dates"""
    df = pd.DataFrame(
        data={
            "headline": [a["fields"]["headline"] for a in articles],
            "date": [a["webPublicationDate"] for a in articles],
        }
    ).sort_values("date", ascending=False)
    return df


"""
TODO:

CRUNCHBASE
~ Will probably be noisy, as the descriptions are short

GTR PROJECTS / CRUNCHBASE
- x Organisational network

HANSARD / GUARDIAN
- x Perhaps of value to check also months when there is more mentions about particular topics?
- x Find words or phrases associated with positive and negative sentiments
- x Sentiment over time
- x Maybe check also how others have done this

- Sentiment of the headline or trailText

~~~~~~~~
Prelim analysis for detecting green documents
- Use my clustering algo with small kNN (perhaps = 5)
- Select clusters that are link to preselected Green topics
- TF_IDF vectorise and train the model
++ Organisational networks

~~~~~~~~

"""
