"""Tests for scoring/metrics computation."""

import numpy as np


def compute_metrics(y_true, predictions, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_pred = (predictions > threshold).astype(int).ravel()
    y_true = y_true.ravel()
    return {
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


class TestScoring:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0])
        preds = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
        m = compute_metrics(y_true, preds)
        assert m["f1_score"] == 1.0
        assert m["accuracy"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        preds = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_metrics(y_true, preds)
        assert m["f1_score"] == 0.0
        assert m["accuracy"] == 0.0

    def test_threshold_sensitivity(self):
        y_true = np.array([1, 1, 0, 0])
        preds = np.array([0.6, 0.4, 0.3, 0.2])
        m_low = compute_metrics(y_true, preds, threshold=0.3)
        m_high = compute_metrics(y_true, preds, threshold=0.5)
        assert m_low["recall"] >= m_high["recall"]

    def test_promotion_decision_positive(self):
        challenger = {"f1_score": 0.90}
        champion = {"f1_score": 0.85}
        should_promote = challenger["f1_score"] > champion["f1_score"] + 0.0
        assert should_promote is True

    def test_promotion_decision_below_threshold(self):
        challenger = {"f1_score": 0.86}
        champion = {"f1_score": 0.85}
        min_improvement = 0.02
        should_promote = challenger["f1_score"] > champion["f1_score"] + min_improvement
        assert should_promote is False
