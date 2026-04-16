"""
evaluate.py

Loads the trained Iris model and evaluates its performance
on the test set.
"""

import logging
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_PATH = os.path.join("models", "iris_model.joblib")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_model(path=MODEL_PATH):
    """Load the trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Please run train.py first."
        )
    logger.info("Loading model from: %s", path)
    model = joblib.load(path)
    logger.info("Model loaded successfully.")
    return model


def load_test_data():
    """Load the Iris test dataset (same split as training)."""
    logger.info("Loading test data...")
    iris = load_iris()
    _, x_test, _, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    logger.info("Test data loaded. Test size: %d", len(x_test))
    return x_test, y_test


def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return the accuracy score."""
    logger.info("Evaluating model...")
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info("Accuracy: %.4f", accuracy)

    # classification_report shows precision, recall, F1 for each class
    labels = sorted(set(y_test) | set(predictions))

    if len(labels) == 3:
        report = classification_report(
            y_test,
            predictions,
            target_names=["setosa", "versicolor", "virginica"]
        )
    else:
        # When not all classes are present, pass explicit labels so
        report = classification_report(y_test, predictions, labels=labels)

    logger.info("Classification Report:\n%s", report)
    return accuracy


def main():
    """Main function: orchestrates model loading and evaluation."""
    model = load_model()
    x_test, y_test = load_test_data()
    accuracy = evaluate_model(model, x_test, y_test)
    logger.info("Evaluation complete. Final accuracy: %.4f", accuracy)
    return accuracy


if __name__ == "__main__":
    main()