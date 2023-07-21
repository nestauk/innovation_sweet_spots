"""
Train model on labelled companies and evaluate on test set

Usage:
    python train_model.py
"""

import typer
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.analysis.notebooks.y2023_childcare import utils
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots.analysis.notebooks.review_labelling.utils import (
    load_training_args,
    load_model,
    load_trainer,
    compute_metrics,
)

SAVE_TRAINING_RESULTS_PATH = PROJECT_DIR / "outputs/2023_childcare/model/"


def train_model():
    ## Load data
    train_ds = load_pickle(utils.SAVE_DS_PATH / "train_ds.pickle")
    test_ds = load_pickle(utils.SAVE_DS_PATH / "test_ds.pickle")
    to_review_ds = load_pickle(utils.SAVE_DS_PATH / "to_review_ds.pickle")

    # Set number of labels
    NUM_LABELS = len(train_ds[0]["labels"])
    NUM_LABELS

    # Path to save intermediary training results and best model
    SAVE_TRAINING_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(num_labels=NUM_LABELS)

    # Train model with early stopping
    training_args = load_training_args(output_dir=SAVE_TRAINING_RESULTS_PATH)
    trainer = load_trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds
    )
    trainer.train()

    # Evaluate model
    trainer.evaluate()

    # View f1, roc and accuracy of predictions on validation set
    predictions = trainer.predict(test_ds)
    compute_metrics(predictions)


if __name__ == "__main__":
    typer.run(train_model)
