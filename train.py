"""
train.py

Trains a Logistic Regression model on the Iris dataset
and saves the trained model to disk.
"""

import logging
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Putting constants at the top makes them easy to find and change.
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.joblib")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Load and split the Iris dataset."""
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    logger.info(
        "Data loaded. Train size: %d, Test size: %d",
        len(x_train), len(x_test)
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Train a Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    logger.info("Model training complete.")
    return model


def save_model(model, path=MODEL_PATH):
    """Save the trained model to disk using joblib."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to: %s", path)


def main():
    """Main function: orchestrates data loading, training, and saving."""
    x_train, x_test, y_train, y_test = load_data()
    model = train_model(x_train, y_train)
    save_model(model)
    logger.info("Training pipeline complete!")
    return model, x_test, y_test


if __name__ == "__main__":
    main()