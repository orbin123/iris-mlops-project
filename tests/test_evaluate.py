"""
test_evaluate.py

Unit tests for the evaluate.py module.
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from train import load_data, train_model, save_model
from evaluate import load_model, load_test_data, evaluate_model


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model_on_disk(tmp_path):
    """Creates a real trained model and saves it to a temp location."""
    x_train, _, y_train, _ = load_data()
    model = train_model(x_train, y_train)
    model_path = str(tmp_path / "iris_model.joblib")
    save_model(model, model_path)
    return model_path, model


# ============================================================
# TESTS FOR load_model()
# ============================================================

class TestLoadModel:
    """Tests for the model loading function."""

    def test_loads_model_successfully(self, model_on_disk):
        """load_model should return a LogisticRegression object."""
        model_path, _ = model_on_disk
        loaded = load_model(model_path)
        assert isinstance(loaded, LogisticRegression)

    def test_raises_error_if_file_missing(self, tmp_path):
        """load_model should raise FileNotFoundError if model doesn't exist."""
        bad_path = str(tmp_path / "nonexistent.joblib")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model(bad_path)
        # Check that the error message is helpful
        assert "train.py" in str(exc_info.value)

    def test_loaded_model_can_predict(self, model_on_disk):
        """Loaded model should be able to make predictions."""
        model_path, _ = model_on_disk
        loaded = load_model(model_path)
        # Test with a dummy sample (4 features for Iris)
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = loaded.predict(sample)
        assert len(prediction) == 1
        assert prediction[0] in {0, 1, 2}


# ============================================================
# TESTS FOR load_test_data()
# ============================================================

class TestLoadTestData:
    """Tests for the test data loading function."""

    def test_returns_two_arrays(self):
        """load_test_data should return x_test and y_test."""
        result = load_test_data()
        assert len(result) == 2

    def test_correct_test_size(self):
        """Test set should have 30 samples (20% of 150)."""
        x_test, y_test = load_test_data()
        assert len(x_test) == 30
        assert len(y_test) == 30

    def test_correct_features(self):
        """Iris test data should have 4 features."""
        x_test, _ = load_test_data()
        assert x_test.shape[1] == 4


# ============================================================
# TESTS FOR evaluate_model()
# ============================================================

class TestEvaluateModel:
    """Tests for the model evaluation function."""

    def test_returns_accuracy_float(self, model_on_disk):
        """evaluate_model should return a float between 0 and 1."""
        model_path, _ = model_on_disk
        model = load_model(model_path)
        x_test, y_test = load_test_data()
        accuracy = evaluate_model(model, x_test, y_test)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_accuracy_above_threshold(self, model_on_disk):
        """A trained Logistic Regression on Iris should exceed 90% accuracy."""
        model_path, _ = model_on_disk
        model = load_model(model_path)
        x_test, y_test = load_test_data()
        accuracy = evaluate_model(model, x_test, y_test)
        assert accuracy >= 0.90