import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers.models.distilbert.tokenization_distilbert_fast import (
    DistilBertTokenizerFast,
)
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)
from transformers.trainer import Trainer
import transformers
from transformers import (
    EvalPrediction,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from typing import Union
from pathlib import Path
from datasets import Dataset


def create_labels(dataset: Dataset, cols_to_skip: list) -> Dataset:
    """Add labels to the dataset. Labels are a list of 0.0s and 1.0s
    corresponding to the order of the labels in the original dataframe
    that was used to create the dataset. Note the 0.0s and 1.0s must be floats."""
    cols = dataset.column_names
    return dataset.map(
        lambda row: {
            "labels": torch.FloatTensor(
                [(row[col]) for col in cols if col not in cols_to_skip]
            )
        }
    )


def tokenize_dataset(dataset: Dataset, text_column: str) -> Dataset:
    """Tokenize text in dataset"""
    remove_cols = dataset.column_names
    remove_cols.remove("labels")
    tokenizer = load_tokenizer()
    return dataset.map(
        lambda row: tokenizer(row[text_column], truncation=True),
        batched=True,
        remove_columns=remove_cols,
    )


def df_to_hf_ds(
    df: pd.DataFrame, non_label_cols: list = ["text", "id"], text_column: str = "text"
) -> Dataset:
    """Converts a dataframe into a huggingface dataset.
    Adds labels and tokenizes the text.

    Args:
        df: Dataframe to convert into a dataset
        non_label_cols: Columns that are not labels.
            Defaults to ["text", "id"].
        text_column: Column in dataframe that contain text to tokenize.
            Defaults to "text".

    Returns:
        Huggingface dataset
    """
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = create_labels(dataset, cols_to_skip=non_label_cols)
    return tokenize_dataset(dataset, text_column=text_column)


def load_tokenizer() -> DistilBertTokenizerFast:
    """Load multi label classification BERT tokenzier"""
    return AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", problem_type="multi_label_classification"
    )


def load_model(
    num_labels: int, model_path: Union[str, Path] = "distilbert-base-uncased"
) -> DistilBertForSequenceClassification:
    """Loads multi label BERT classifier

    Args:
        num_labels: Number of labels
        model_path: Defaults to "distilbert-base-uncased". Alternatively,
            can specify path to a fine tuned model.

    Returns:
        BERT classifier model
    """
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )


def load_training_args(
    output_dir: Union[str, Path]
) -> transformers.training_args.TrainingArguments:
    """Load Training Arguments to be used to train the model"""
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=30,
        weight_decay=0.01,
        evaluation_strategy="steps",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        eval_steps=50,
    )


def load_trainer(
    model: DistilBertForSequenceClassification,
    args: transformers.training_args.TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    """Load model trainer which can be used to train a model or make predictions"""
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=load_tokenizer(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )


def dummies_to_labels(dummy_cols: pd.DataFrame) -> pd.Series:
    """Return a series with a set of labels that
    correspond to the 1s in the dummy column names"""
    return (
        dummy_cols.stack()
        .loc[lambda x: x == 1]
        .reset_index()
        .groupby("level_0")
        .agg({"level_1": set})
    )


def binarise_predictions(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply sigmoid transformation to predictions and set
    values >= threshold to 1 and < threshold to 0"""
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    binarised = np.zeros(probs.shape)
    binarised[np.where(probs >= threshold)] = 1
    return binarised


def multi_label_metrics(
    predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict:
    """Calculate and return dictionary of metrics that are useful
    for measuring multi label, multi class classification models"""
    y_pred = binarise_predictions(predictions, threshold)
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    return {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}


def compute_metrics(model_predictions: EvalPrediction) -> dict:
    """Compute metrics in format suitable for pytorch training"""
    preds = (
        model_predictions.predictions[0]
        if isinstance(model_predictions.predictions, tuple)
        else model_predictions.predictions
    )
    return multi_label_metrics(predictions=preds, labels=model_predictions.label_ids)
