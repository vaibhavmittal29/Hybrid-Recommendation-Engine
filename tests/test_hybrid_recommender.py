import numpy as np
from src.hybrid_recommender import HybridRecommender


class DummyCF:
    def predict_batch_users(self, user_idx, items):
        return np.ones(len(items)) * 0.5


class DummyCB:
    def predict_batch_users(self, user_idx, items):
        return np.ones(len(items)) * 0.8


def test_sigmoid_blending():
    user_counts = {0: 2, 1: 20}  # 0 is cold, 1 is warm
    hr = HybridRecommender(DummyCF(), DummyCB(), user_counts)

    # Cold user check (n=2, tau=10)
    # alpha should be small, so CB weight should be high
    alpha_cold = hr._sigmoid_weight(2)
    assert alpha_cold < 0.1, "Cold user should have low CF weight"

    # Warm user check (n=20, tau=10)
    # alpha should be near 1
    alpha_warm = hr._sigmoid_weight(20)
    assert alpha_warm > 0.9, "Warm user should have high CF weight"
