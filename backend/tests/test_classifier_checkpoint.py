"""
Test 2: classifier checkpoint loading. Skipped automatically if the
real checkpoint file isn't present -- this project ships no fake
weights to substitute for it.
"""

import os

import pytest

from app.config import CLASSIFIER_CHECKPOINT_PATH


@pytest.mark.skipif(
    not os.path.exists(CLASSIFIER_CHECKPOINT_PATH),
    reason="best_4class_classifier.pt not present at CLASSIFIER_CHECKPOINT_PATH.",
)
def test_classifier_loads_and_predicts():
    import torch

    from app.ml.classifier import Classifier
    from app.ml.feature_extractor import ViewEmbeddings

    clf = Classifier()
    assert len(clf.labels) == 4

    dummy = ViewEmbeddings(
        e_main=torch.randn(768), e_sub=torch.randn(768), e_comments=torch.randn(768)
    )
    prediction = clf.predict(dummy)
    assert prediction.predicted_class in clf.labels
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-4
