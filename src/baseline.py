import numpy as np


class PopularityBaseline:
    def __init__(self, train_df, num_items):
        self.item_counts = train_df["item_idx"].value_counts().to_dict()
        self.pop_scores = np.zeros(num_items)
        for idx, count in self.item_counts.items():
            self.pop_scores[idx] = count

    def predict_batch_users(self, user_idx, all_items):
        # We return the same popularity scores for everyone
        scores = self.pop_scores[all_items]
        # Return 3 elements to match HybridRecommender signature for evaluation
        return scores, scores, scores, 0.0
