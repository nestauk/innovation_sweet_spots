# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    load_training_args,
    load_model,
    load_trainer,
)
from innovation_sweet_spots.utils.io import load_pickle

# %%
# Load datasets
LOAD_DS_PATH = PROJECT_DIR / "inputs/data/review_labelling/datasets/foodtech_gtr/"
train_ds = load_pickle(LOAD_DS_PATH / "train_ds.pickle")
valid_ds = load_pickle(LOAD_DS_PATH / "valid_ds.pickle")
to_review_ds = load_pickle(LOAD_DS_PATH / "to_review_ds.pickle")

# %%
# Set number of labels
NUM_LABELS = len(train_ds[0]["labels"])

# %%
# Path to save intermediary training results and best model
SAVE_TRAINING_RESULTS_PATH = (
    PROJECT_DIR / "outputs/data/review_labelling/models/foodtech_gtr/results/"
)
SAVE_TRAINING_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# %%
# Load model
model = load_model(num_labels=NUM_LABELS)

# %%
# Train model with early stopping
training_args = load_training_args(output_dir=SAVE_TRAINING_RESULTS_PATH)
trainer = load_trainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds
)
trainer.train()

# %%
# Evaluate model
trainer.evaluate()

# %%
# View f1, roc and accuracy of predictions on validation set
predictions = trainer.predict(valid_ds)
compute_metrics(predictions)

# %%
