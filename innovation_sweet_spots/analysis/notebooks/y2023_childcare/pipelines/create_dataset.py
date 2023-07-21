"""
This script creates the dataset for the company relevancy model.

Usage:
    python create_dataset.py
"""
import typer
from innovation_sweet_spots.analysis.notebooks.y2023_childcare import utils


def create_dataset():
    utils.prepare_training_data()


if __name__ == "__main__":
    typer.run(create_dataset)
