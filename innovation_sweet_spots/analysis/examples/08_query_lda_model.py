# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Using `lda_modelling_utils`

# +
import pandas as pd
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.getters.models
from innovation_sweet_spots.analysis import lda_modelling_utils as lmu
from innovation_sweet_spots.analysis.wrangling_utils import GtrWrangler

GTR = GtrWrangler()
MODEL_DIR = innovation_sweet_spots.getters.models.PILOT_MODELS_DIR / "lda"
# -

# ## Saving a topic model
#
# Example of saving a model once it's been trained.

# +
# Initialise a test model
test_model = lmu.tp.LDAModel(k=2)
# Generate some dummy data
dummy_data = pd.DataFrame(
    data={"id": [str(i) for i in range(10)], "text": [["test", "doc"]] * 10}
)
# Train the test model with dummy data
for doc in dummy_data.text.to_list():
    test_model.add_doc(doc)
test_model.train(10)

# Example how to save the trained model
lmu.save_lda_model_data(
    model_name="test_model",
    folder=MODEL_DIR,
    topic_model=test_model,
    document_ids=dummy_data.id.to_list(),
)
# -

# ## Loading and a trained topic model
#
# You can use function `load_lda_model_data()` to load in a topic model and its associated data.

# Load in the model that we just saved
mdl = lmu.load_lda_model_data(model_name="test_model", folder=MODEL_DIR)

type(mdl)

mdl.keys()

# Alternatively, to load in the models used in the pilot project, you can use `getters`:

# Specify the model name from pilot project
MODEL_NAME = "lda_broad_corpus"
# Load the model (might take a minute for a large model)
model_data = innovation_sweet_spots.getters.models.get_pilot_lda_model(MODEL_NAME)

# Print some basic info about the model
lmu.print_model_info(model_data["model"])

# Check the topic description table
model_data["topic_descriptions"].head(10)

# Save a visualisation of the trained topic model
lmu.make_pyLDAvis(model_data["model"], MODEL_DIR / f"{MODEL_NAME}_pyLDAvis.html")

# ## Find the most typical documents for a given topic

Query = lmu.QueryTopics(model_data)

# Choose a topic
TOPIC = 71

# Sort document ids in terms of their topic probabilities
sorted_docs = Query.sort_documents_by_topic(topic_id=TOPIC)
sorted_docs.head(5)

# Add project data to these documents
# Note: the topic model in the pilot project was trained on both GtR and Crunchbase data;
# in the output below, the null values correspond to Crunchbase documents
GTR.add_project_data(sorted_docs, id_column="id", columns=["title"]).head(
    20
).title.to_list()

# ## Another example: Loading a topic model's document probability matrix
#
# Example where we've inferred document topic probabilities for a set of documents, many of which were not used to train the model.
#
# In this case, their topic probabilities are stored in a `.npy` file, whereas the document identifiers (corresponding to the first dimension of the topic probability array) are stored in a separate `.txt` file.
#
# The function `load_document_probabilities()` loads in both files and stores them in a dictionary.

MODEL_NAME = "lda_narrow_corpus"

# Load the model
model_data = innovation_sweet_spots.getters.models.get_pilot_lda_model(MODEL_NAME)

# Print some basic info
lmu.print_model_info(model_data["model"])

# Output a visualisation of the trained topic model
lmu.make_pyLDAvis(model_data["model"], MODEL_DIR / f"{MODEL_NAME}_pyLDAvis.html")

doc_probs = lmu.load_document_probabilities(
    model_name="lda_narrow_corpus", folder=MODEL_DIR
)
Query = lmu.QueryTopics(doc_probs)

doc_probs.keys()

Query.topic_descriptions

# Sort document ids in terms of their topic probabilities
sorted_docs = Query.sort_documents_by_topic(topic_id=146)
sorted_docs.head(5)

# Add project data to these documents
# Note: the topic model in the pilot project was trained on both GtR and Crunchbase data;
# in the output below, the null values correspond to Crunchbase documents
GTR.add_project_data(sorted_docs, id_column="id", columns=["title", "start"]).head(20)[
    ["title", "start"]
]
