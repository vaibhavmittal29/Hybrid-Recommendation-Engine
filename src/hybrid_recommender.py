import numpy as np
from src.config import SIGMOID_TAU, SIGMOID_BETA


class HybridRecommender:
    def __init__(self, cf_model, cb_model, user_counts):
        self.cf = cf_model
        self.cb = cb_model
        self.user_counts = user_counts

    def _sigmoid_weight(self, n_u):
        return 1.0 / (1.0 + np.exp(-SIGMOID_BETA * (n_u - SIGMOID_TAU)))

    def predict_batch_users(self, user_idx, all_items):
        cf_scores = self.cf.predict_batch_users(user_idx, all_items)
        cb_scores = self.cb.predict_batch_users(user_idx, all_items)

        def min_max_scale(scores):
            min_s = scores.min()
            max_s = scores.max()
            if max_s > min_s:
                return (scores - min_s) / (max_s - min_s)
            return np.zeros_like(scores)

        cf_norm = min_max_scale(cf_scores)
        cb_norm = min_max_scale(cb_scores)

        n_u = self.user_counts.get(user_idx, 0)
        alpha = self._sigmoid_weight(n_u)

        hybrid_scores = alpha * cf_norm + (1.0 - alpha) * cb_norm

        return hybrid_scores, cf_norm, cb_norm, alpha
