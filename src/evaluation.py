import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import RATING_THRESHOLD, K_LIST, NUM_NEGATIVE_CANDIDATES


def precision_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    if len(relevant) == 0:
        return 0.0
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


class Evaluator:
    def __init__(
        self, recommender, train_df, test_df, num_items, k_list=K_LIST, reranker=None
    ):
        self.recommender = recommender
        self.k_list = k_list
        self.reranker = reranker

        # Build user history (all items a user has interacted with)
        self.user_history = {}
        for user_idx, group in pd.concat([train_df, test_df]).groupby("user_idx"):
            self.user_history[user_idx] = set(group["item_idx"].tolist())

        # Build ground truth (relevant items in test set)
        self.user_true_items = {}
        for user_idx, group in test_df.groupby("user_idx"):
            relevant = group[group["rating"] >= RATING_THRESHOLD]["item_idx"].tolist()
            if len(relevant) > 0:
                self.user_true_items[user_idx] = set(relevant)

        self.num_items = num_items
        self.all_items_set = set(range(num_items))

    def _sample_negatives(self, user_idx, num_negatives=NUM_NEGATIVE_CANDIDATES):
        history = self.user_history.get(user_idx, set())
        available_negatives = list(self.all_items_set - history)

        if len(available_negatives) > num_negatives:
            return np.random.choice(
                available_negatives, size=num_negatives, replace=False
            )
        return np.array(available_negatives)

    def evaluate_users(self, user_indices):
        metrics = {f"Precision@{k}": [] for k in self.k_list}
        metrics.update({f"Recall@{k}": [] for k in self.k_list})
        metrics.update({f"NDCG@{k}": [] for k in self.k_list})

        for u in tqdm(user_indices, desc="Evaluating"):
            if u not in self.user_true_items:
                continue

            relevant = self.user_true_items[u]

            # Form candidate set: Relevant items + 500 sampled negatives
            negatives = self._sample_negatives(u)
            candidates = np.array(list(relevant) + list(negatives))

            # Predict
            scores, _, _, _ = self.recommender.predict_batch_users(u, candidates)

            if self.reranker is not None:
                # Need enough candidates for top K
                max_k = max(self.k_list)
                # First get top 2*max_k items by score to feed into MMR
                # This ensures MMR has a pool of good items to select from
                pool_size = min(len(candidates), 5 * max_k)
                pool_indices = np.argsort(scores)[::-1][:pool_size]

                pool_items = candidates[pool_indices]
                pool_scores = scores[pool_indices]

                top_items = self.reranker.rerank(
                    u, pool_items, pool_scores, top_k=max_k
                )
            else:
                max_k = max(self.k_list)
                top_idx = np.argsort(scores)[::-1][:max_k]
                top_items = candidates[top_idx]

            for k in self.k_list:
                metrics[f"Precision@{k}"].append(precision_at_k(top_items, relevant, k))
                metrics[f"Recall@{k}"].append(recall_at_k(top_items, relevant, k))
                metrics[f"NDCG@{k}"].append(ndcg_at_k(top_items, relevant, k))

        agg_metrics = {
            m: np.mean(vals) if len(vals) > 0 else 0.0 for m, vals in metrics.items()
        }
        return agg_metrics

    def evaluate_slice(self, users_set):
        eval_users = [u for u in users_set if u in self.user_true_items]
        if not eval_users:
            return None
        return self.evaluate_users(eval_users)
