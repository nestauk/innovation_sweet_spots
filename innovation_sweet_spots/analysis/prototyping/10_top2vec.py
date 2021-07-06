# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.utils.io as iss_io
import innovation_sweet_spots.utils.text_cleaning_utils as iss_text
import innovation_sweet_spots.analysis.text_analysis as iss_text_analysis
from innovation_sweet_spots.getters import gtr, misc

DATA_OUTPUTS = PROJECT_DIR / "outputs/data"


# %%
import numpy as np
import pandas as pd
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import gensim.models.phrases as phrases

# %%
green_proj = iss_io.read_list_of_terms(DATA_OUTPUTS / "gtr/green_gtr_project_ids.txt")

# %%
gtr_projects = gtr.get_gtr_projects()

# %%
gtr_projects_green = gtr_projects[gtr_projects.project_id.isin(green_proj)]

# %%
gtr_projects_green.info()

# %%
gtr_projects_green[["project_id", "title", "abstractText"]]

# %%
import importlib

importlib.reload(iss_text)

# %%
green_documents = iss.create_documents_from_dataframe(
    gtr_projects_green,
    columns=["title", "abstractText", "techAbstractText"],
    preprocessor=(lambda x: x),
)

# %%
i = np.random.randint(len(green_documents))
print(i)
green_documents[i]

# %%
iss_text.clean_up(iss_text.clean_punctuation(green_documents[i]))

# %%

# %%
# # nlp = iss_text_analysis.setup_spacy_model()
# text = iss_text.clean_up(iss_text.clean_punctuation(green_documents[i]))
# doc = nlp(text)
# sentences = [str(sent) for sent in doc.sents]

# %%
# text = green_documents[i]
# texts = [iss_text.lowercase(iss_text.clean_up(iss_text.clean_punctuation(green_documents[i]))) for i in [0,1,3]]

# %%
docs = [gensim.utils.simple_preprocess(doc, deacc=True) for doc in green_documents]

# %%
# sentence_stream = [doc.split(" ") for doc in texts]

# %%
docs = [gensim.utils.simple_preprocess(doc, deacc=True) for doc in green_documents]

# %%
bigram = Phrases(
    docs,
    min_count=3,
    threshold=0.33,
    delimiter="_",
    scoring="npmi",
    connector_words=phrases.ENGLISH_CONNECTOR_WORDS,
)
bigram_phraser = Phraser(bigram)

# %%
# trigram = Phrases(bigram[sentence_stream], min_count=2, threshold=0.5, delimiter='_', scoring="npmi", connector_words=phrases.ENGLISH_CONNECTOR_WORDS)

# %%
# trigram_phraser = Phraser(trigram)

# %%
p = bigram_phraser.find_phrases(docs)
df = pd.DataFrame(data={"phrase": p.keys(), "score": p.values()})

# %%
# df[df.phrase.str.contains('_heat')]

# %%
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags


# %%
def bigrammer(doc):
    # sentence_stream = doc.split(" ")
    sentence_stream = gensim.utils.simple_preprocess(doc, deacc=True)
    return bigram_phraser[sentence_stream]


# %%
# bigrammer('Climate change: Poor households in UK should get free heat pumps, say experts. Help is needed to replace gas boilers with low-carbon alternatives, warn builders, energy firms and charities')

# %%
len(gtr_projects_green)

# %%
df = pd.DataFrame(
    data={
        "project_id": gtr_projects_green.project_id.to_list(),
        "title": gtr_projects_green.title.to_list(),
        "text": green_documents,
    }
)


# %%
df.to_csv("green_docs.csv", index=False)

# %%
import wikipedia

# %%
from wiki_topic_labels import suggest_labels

# %%
topic = [
    "ocean_acidification",
    "ecosystem_function",
    "ecosystem_functioning",
    "organisms",
    "tolerant",
    "marine_environment",
]
topic = [
    "wind_turbine",
    "solar_cells",
    "grid",
    "offshore_wind",
    "electrolyte",
    "co_capture",
]

# %%
topix = [" ".join(t.split("_")) for t in topic]
topix

# %%
suggest_labels(topix)

# %%
suggest_labels([topix[5]])

# %% [markdown]
# # Top2vec

# %%
import top2vec

# %%
