import pandas as pd
import nltk.data
import nltk
from innovation_sweet_spots.analysis import analysis_utils as au
from innovation_sweet_spots.utils import chart_trends

# # Variables for colour scale (might be useful)
# colour_domain = [
#     "Health",
#     "Innovative food",
#     "Logistics",
#     "Restaurants and retail",
#     "Cooking and kitchen",
#     "Food waste",
# ]
# colour_range_ = pu.NESTA_COLOURS[0 : len(domain)]

nltk.download("punkt")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def remove_space_after_comma(text):
    """util function to process search terms with comma"""
    return ",".join([s.strip() for s in text.split(",")])


def check_articles_for_comma_terms(text: str, terms: str):
    terms = [term.strip() for term in terms.split(",")]
    sentences_with_terms = find_sentences_with_terms(text, terms, all_terms=True)
    return len(sentences_with_terms) >= 1


def find_sentences_with_terms(text, terms, all_terms: bool = True):
    """util function that finds terms in sentences"""
    # split text into sentences
    sentences = tokenizer.tokenize(text)
    # keep sentences with terms
    sentences_with_terms = []
    # number of terms in the query
    n_terms = len(terms)
    for sentence in sentences:
        terms_detected = 0
        # check all terms
        for term in terms:
            if term in sentence.lower():
                terms_detected += 1
        # check if all terms were found
        if (
            all_terms
            and terms_detected == n_terms
            or not all_terms
            and terms_detected > 0
        ):
            sentences_with_terms.append(sentence)
    return sentences_with_terms


def get_ts(df_id_to_term, df_baseline, category="Category"):
    """build time series"""
    return (
        df_id_to_term.drop_duplicates(["id", category], keep="first")
        .groupby(["year", category])
        .agg(counts=("id", "count"))
        .reset_index()
        .merge(
            df_baseline.rename(columns={"counts": "total_counts"}),
            on="year",
            how="left",
        )
        .assign(fraction=lambda df: df.counts / df.total_counts)
    )


def get_magnitude_growth(
    ts: pd.DataFrame, variable: str, category: str = "Category", verbose: bool = True
):
    """Calculate magnitude and growth"""
    categories_to_check = ts[category].unique()
    magnitude_growth = []
    # Go throuh each unique category
    for tech_area in categories_to_check:
        if verbose:
            print(tech_area)
        df = ts.query(f"`{category}` == @tech_area").drop(category, axis=1)[
            ["year", variable]
        ]
        df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
        magnitude_growth.append(
            [
                df_trends.query('trend == "magnitude"').iloc[0][variable],
                df_trends.query('trend == "growth"').iloc[0][variable],
                tech_area,
            ]
        )
    magnitude_growth_df = pd.DataFrame(
        magnitude_growth, columns=["magnitude", "growth", category]
    ).assign(growth=lambda df: df.growth / 100)
    return chart_trends.estimate_trend_type(
        magnitude_growth_df, magnitude_column="magnitude", growth_column="growth"
    )
