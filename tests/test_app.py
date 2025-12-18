import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import predict


def test_predict_returns_valid_class_id():
    features = [5.1, 3.5, 1.4, 0.2]

    class_id = predict(features)

    assert class_id in (0, 1, 2)
