"""
test_integration.py

Integration tests: test the full pipeline end-to-end.
Mocking tests: test functions in isolation using fake objects.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from train import load_data, train_model, save_model
from evaluate import load_model, load_test_data, evaluate_model

# Integration Tests
class TestFullPipeline:
    """End-to-end tests for the complete ML pipeline."""

    def test_full_train_evaluate_pipeline(self, tmp_path):
        """
        INTEGRATION TEST: Run the full pipeline from data loading to evaluation.
        
        This is the most important test. It simulates exactly what happens
        in production: train a model, save it, load it, evaluate it.
        """
        # --- STEP 1: TRAIN ---
        x_train, x_test, y_train, y_test = load_data()
        model = train_model(x_train, y_train)
        
        # --- STEP 2: SAVE ---
        model_path = str(tmp_path / "pipeline_model.joblib")
        save_model(model, model_path)
        assert os.path.exists(model_path), "Model file was not created"
        
        # --- STEP 3: LOAD ---
        loaded_model = load_model(model_path)
        assert loaded_model is not None, "Model could not be loaded"
        
        # --- STEP 4: EVALUATE ---
        accuracy = evaluate_model(loaded_model, x_test, y_test)
        
        # --- ASSERT ---
        assert accuracy >= 0.90, (
            f"Pipeline accuracy {accuracy:.2f} is below acceptable threshold of 90%"
        )

    def test_pipeline_is_deterministic(self, tmp_path):
        """
        The pipeline should produce the same results every run.
        
        Why? Because we use random_state=42. This is critical in ML:
        reproducibility means you can debug and compare experiments.
        """
        results = []
        
        for _ in range(2):
            x_train, x_test, y_train, y_test = load_data()
            model = train_model(x_train, y_train)
            accuracy = evaluate_model(model, x_test, y_test)
            results.append(accuracy)
        
        assert results[0] == results[1], (
            f"Pipeline is not deterministic: {results[0]} != {results[1]}"
        )

    def test_model_generalizes_to_new_samples(self, tmp_path):
        """
        The model should make sensible predictions on known Iris samples.
        
        Real-world test: setosa has small petals, so features like
        [5.1, 3.5, 1.4, 0.2] should map to class 0 (setosa).
        """
        x_train, _, y_train, _ = load_data()
        model = train_model(x_train, y_train)
        
        # These are real Iris samples with known labels
        known_setosa = np.array([[5.1, 3.5, 1.4, 0.2]])  # Class 0
        known_virginica = np.array([[6.3, 3.3, 6.0, 2.5]])  # Class 2
        
        pred_setosa = model.predict(known_setosa)[0]
        pred_virginica = model.predict(known_virginica)[0]
        
        assert pred_setosa == 0, f"Expected setosa (0), got {pred_setosa}"
        assert pred_virginica == 2, f"Expected virginica (2), got {pred_virginica}"

# Mocking Tests
class TestMocking:
    """Tests that use mocking to isolate functions from dependencies."""

    def test_load_model_calls_joblib(self, tmp_path):
        """
        MOCK TEST: Verify load_model uses joblib.load internally.
        
        We use patch() to replace joblib.load with a fake function.
        This lets us test load_model without needing a real file.
        """
        # Create a mock model object that behaves like a real model
        mock_model = MagicMock(spec=["predict", "score"])
        mock_model.predict.return_value = np.array([0, 1, 2])
        
        # Create a real file so the 'file exists' check passes
        fake_model_path = str(tmp_path / "fake_model.joblib")
        with open(fake_model_path, "w") as f:
            f.write("fake")
        
        # patch() intercepts the call to joblib.load and returns our mock
        with patch("evaluate.joblib.load", return_value=mock_model) as mock_load:
            result = load_model(fake_model_path)
            
            # Verify joblib.load was called exactly once
            mock_load.assert_called_once_with(fake_model_path)
            
            # Verify we got back our mock model
            assert result == mock_model

    def test_save_model_calls_joblib_dump(self, tmp_path):
        """
        MOCK TEST: Verify save_model uses joblib.dump correctly.
        """
        fake_model = MagicMock()
        model_path = str(tmp_path / "output_model.joblib")
        
        with patch("train.joblib.dump") as mock_dump:
            save_model(fake_model, model_path)
            
            # Verify joblib.dump was called with the right arguments
            mock_dump.assert_called_once_with(fake_model, model_path)

    def test_evaluate_with_perfect_predictions(self):
        """
        MOCK TEST: Test evaluate_model logic with a perfectly accurate model.
        
        By using a mock, we control exactly what the model predicts,
        letting us test the evaluation logic in isolation.
        """
        # Create a mock model that always predicts perfectly
        mock_model = MagicMock()
        y_test = np.array([0, 1, 2, 0, 1, 2])
        mock_model.predict.return_value = y_test  # Perfect predictions
        
        x_test = np.zeros((6, 4))  # Dummy features (not used, mock ignores them)
        
        accuracy = evaluate_model(mock_model, x_test, y_test)
        
        assert accuracy == 1.0, f"Expected perfect accuracy 1.0, got {accuracy}"
        mock_model.predict.assert_called_once_with(x_test)

    def test_evaluate_with_zero_accuracy(self):
        """
        MOCK TEST: Test evaluate_model with a completely wrong model.
        """
        mock_model = MagicMock()
        y_test = np.array([0, 0, 0])
        mock_model.predict.return_value = np.array([1, 1, 1])  # Always wrong
        
        x_test = np.zeros((3, 4))
        accuracy = evaluate_model(mock_model, x_test, y_test)
        
        assert accuracy == 0.0, f"Expected zero accuracy, got {accuracy}"