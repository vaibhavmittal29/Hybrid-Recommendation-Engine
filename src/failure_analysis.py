import numpy as np


class FailureAnalyzer:
    def __init__(self, recommender, train_df, test_df, movies_df, evaluator):
        self.recommender = recommender
        self.train_df = train_df
        self.test_df = test_df
        self.movies_df = movies_df
        self.evaluator = evaluator

    def analyze(self, user_indices, report_path):
        failures = []
        for u in user_indices:
            if u not in self.evaluator.user_true_items:
                continue

            relevant = self.evaluator.user_true_items[u]
            negatives = self.evaluator._sample_negatives(u)
            candidates = np.array(list(relevant) + list(negatives))

            scores, _, _, _ = self.recommender.predict_batch_users(u, candidates)
            top_idx = np.argsort(scores)[::-1][:10]
            top_items = candidates[top_idx]

            ndcg = self._ndcg_at_k(top_items, relevant, 10)
            if ndcg == 0.0:
                hist_len = len(self.train_df[self.train_df["user_idx"] == u])
                failures.append(
                    {
                        "user": u,
                        "history_len": hist_len,
                        "relevant": relevant,
                        "top_items": top_items,
                    }
                )

        # Categorize
        categories = {
            "sparse_history": 0,
            "metadata_poverty": 0,
            "popularity_bias": 0,
            "other": 0,
        }

        with open(report_path, "w") as f:
            f.write("# Failure Analysis Details\n\n")
            f.write(f"Total failures (NDCG@10 = 0): {len(failures)}\n\n")

            for fail in failures[:20]:  # Report up to 20 detailed examples
                u = fail["user"]
                h_len = fail["history_len"]
                f.write(f"### User {u} (History Length: {h_len})\n")

                if h_len < 5:
                    cat = "sparse_history"
                else:
                    cat = "other"
                categories[cat] += 1

                f.write(f"**Category:** {cat}\n")
                f.write("**Relevant Items (Missed):**\n")
                for item in list(fail["relevant"])[:5]:
                    title = self.movies_df[self.movies_df["item_idx"] == item][
                        "title"
                    ].values[0]
                    f.write(f"- {title}\n")

                f.write("**Top 5 Recommended:**\n")
                for item in list(fail["top_items"])[:5]:
                    title = self.movies_df[self.movies_df["item_idx"] == item][
                        "title"
                    ].values[0]
                    f.write(f"- {title}\n")
                f.write("\n---\n\n")

            f.write("## Summary of Categories\n")
            for k, v in categories.items():
                f.write(f"- {k}: {v}\n")

    def _ndcg_at_k(self, recommended, relevant, k):
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, item in enumerate(recommended[:k])
            if item in relevant
        )
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0
