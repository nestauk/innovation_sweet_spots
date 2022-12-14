# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis import analysis_utils as au
import utils
import altair as alt
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import re
from nltk.corpus import stopwords


# %%
def sentence_to_lemmas(sentence):
    if type(sentence) is str:
        return [
            lemmatizer.lemmatize(re.sub(r"\W+", "", w))
            for w in sentence.lower().split()
        ]
    else:
        return [""]


# %%
term = "food delivery"

# %%
utils.find_sentences_with_terms("bbbb reformulation", term)

# %%
df = pd.read_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/guardian_articles_delivery.csv"
)


# %%
def get_sentences(df, term):
    df_text = df[["id", "year", "text"]].copy()
    df_text["sentences"] = df_text.text.apply(
        lambda x: utils.find_sentences_with_terms(x, [term])
    )
    df_sentences = df_text.explode("sentences")
    df_sentences["lemmas"] = df_sentences["sentences"].apply(sentence_to_lemmas)
    df_sentences["sentence_id"] = list(range(0, len(df_sentences)))
    return df_sentences


def get_all_sentences(df):
    df_text = df[["id", "year", "text"]].copy()
    df_text["sentences"] = df_text.text.apply(lambda x: utils.tokenizer.tokenize(x))
    df_sentences = df_text.explode("sentences")
    df_sentences["lemmas"] = df_sentences["sentences"].apply(sentence_to_lemmas)
    df_sentences["sentence_id"] = list(range(0, len(df_sentences)))
    return df_sentences


def get_lemmas(df_sentences):
    return (
        df_sentences[["id", "year", "lemmas", "sentence_id"]]
        .explode("lemmas")
        .assign(count=1)
    )


def filtered_lemmas(df_lemmas):
    df_lemmas_included = (
        df_lemmas.groupby(["lemmas"], as_index=False)
        .agg(counts=("count", "count"))
        .query("counts > 10")
        .query("lemmas not in @stopwords.words('english')")
        .query("lemmas != ''")
    )
    return df_lemmas_included


def get_lemmas_ts(df_lemmas, df_lemmas_included):
    return (
        df_lemmas.groupby(["lemmas", "year"], as_index=False)
        .count()
        .astype({"year": str})
        .query("lemmas in @df_lemmas_included.lemmas.to_list()")
    )


def plot_mentions(df_lemmas_ts, co_term):
    df_ts = au.impute_empty_periods(
        df_lemmas_ts.query("lemmas == @co_term").assign(
            period=lambda df: pd.to_datetime(df.year)
        ),
        "period",
        "Y",
        2000,
        2021,
    ).assign(year=lambda df: df.period.dt.year)
    return (
        alt.Chart(df_ts)
        .mark_line()
        .encode(x="year", y="count", tooltip=["year", "count"])
    )


# %%
df_sentences = get_all_sentences(df)
df_lemmas = get_lemmas(df_sentences)
ok_lemmas = filtered_lemmas(df_lemmas)
lemmas_ts = get_lemmas_ts(df_lemmas, ok_lemmas)

# %%
ok_lemmas.sort_values("counts", ascending=False).head(20)

# %%
plot_mentions(lemmas_ts, "sugar")

# %%
plot_mentions(lemmas_ts, "fibre")

# %%
plot_mentions(lemmas_ts, "tax")

# %%
df_lemmas.query("lemmas == 'fibre'")

# %%
df_sentences.query("sentence_id == 1036").sentences.iloc[0]

# %%
df_sentences = get_sentences(df, term)
df_lemmas = get_lemmas(df_sentences)
ok_lemmas = filtered_lemmas(df_lemmas)
lemmas_ts = get_lemmas_ts(df_lemmas, ok_lemmas)

# %%
ok_lemmas.sort_values("counts", ascending=False).head(20)

# %%
plot_mentions(lemmas_ts, "deliveroo")

# %%
