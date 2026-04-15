"""
test_train.py

Unit tests for the train.py module.
Tests each function in isolation.
"""

import os 
import pytest 
import joblib 
import numpy as np
from sklearn.linear_model import LogisticRegression

from train import load_data, train_model, save_model, MODEL_PATH

# Fixtures
@pytest.fixture 
def sample_data():
    """Provides a small training dataset for tests"""
    x_train, x_test, y_train, y_test = load_data()
    return x_train, x_test, y_train, y_test

@pytest.fixture 
def trained_model(sample_data):
    """Provides a trained model for tests"""
    x_train, _, y_train, _ = sample_data
    model = train_model(x_train, y_train)
    return model 

# Tests for load_data()

class TestLoadData:
    """Tests for the data loading function"""

    def test_returns_split_sizes(self, sample_data):
        """Training set should be 80% and test set 20% of 150 samples"""
        x_train, x_test, y_train, y_test = sample_data
        assert len(x_train) == 120 
        assert len(x_test) == 30
        assert len(y_train) == 120
        assert len(y_test) == 30

    def test_correct_feature_count(self, sample_data):
        """Iris dataset has 4 features per sample."""
        x_train, x_test, _, _ = sample_data
        assert x_train.shape[1] == 4
        assert x_test.shape[1] == 4

    def test_labels_are_valid(self, sample_data):
        """Labels should only be 0, 1, or 2 (three Iris classes)."""
        _, _, y_train, y_test = sample_data
        all_labels = np.concatenate([y_train, y_test])
        assert set(all_labels).issubset({0, 1, 2})

    def test_no_missing_values(self, sample_data):
        """There should be no NaN values in the data."""
        x_train, x_test, _, _ = sample_data
        assert not np.isnan(x_train).any()
        assert not np.isnan(x_test).any()

# Tests for train_model()

class TestTrainModel():
    """Tests for the model training functions"""

    def test_returns_logistic_regression(self, sample_data, trained_model):
        """train_model should return a Logistic Regression object"""
        assert isinstance(trained_model, LogisticRegression)
    
    def test_model_can_predict(self, sample_data, trained_model):
        """Trained model should be able to make predictions"""
        _, x_test, _, _ = sample_data 
        predictions = trained_model.predict(x_test)
        assert len(predictions) == 30

    def test_predictions_are_valid_classes(self, sample_data, trained_model):
        """All predictions should be valid class labels."""
        _, x_test, _, _ = sample_data
        predictions = trained_model.predict(x_test)
        assert set(predictions).issubset({0, 1, 2})

    def test_model_accuracy_above_threshold(self, sample_data, trained_model):
        """A well-trained model should achieve at least 90% accuracy."""
        _, x_test, _, y_test = sample_data
        accuracy = trained_model.score(x_test, y_test)
        assert accuracy >= 0.90, f"Accuracy {accuracy:.2f} is below 90%"

class TestSaveModel:
    """Tests for the model saving function."""

    def test_model_file_is_created(self, trained_model, tmp_path):
        """save_model should create a file at the specified path."""
        # tmp_path is a special pytest fixture that gives a temporary folder
        test_path = tmp_path / "test_model.joblib"
        save_model(trained_model, str(test_path))
        assert test_path.exists()

    def test_saved_model_can_be_loaded(self, trained_model, tmp_path):
        """A saved model should be loadable and usable."""
        test_path = tmp_path / "test_model.joblib"
        save_model(trained_model, str(test_path))

        loaded_model = joblib.load(str(test_path))
        assert isinstance(loaded_model, LogisticRegression)

    def test_loaded_model_gives_same_predictions(self, sample_data, trained_model, tmp_path):
        """Loaded model should give identical predictions to the original."""
        _, x_test, _, _ = sample_data
        test_path = tmp_path / "test_model.joblib"
        save_model(trained_model, str(test_path))

        loaded_model = joblib.load(str(test_path))
        original_preds = trained_model.predict(x_test)
        loaded_preds = loaded_model.predict(x_test)

        np.testing.assert_array_equal(original_preds, loaded_preds)