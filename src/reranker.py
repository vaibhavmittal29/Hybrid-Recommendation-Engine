import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import MMR_LAMBDA


class MMRReranker:
    def __init__(self, tfidf_matrix, lambda_param=MMR_LAMBDA):
        self.tfidf_matrix = tfidf_matrix
        self.lambda_param = lambda_param

    def rerank(self, user_idx, candidate_items, candidate_scores, top_k):
        """
        Rerank a set of candidate items using MMR.
        candidate_items: list/array of item indices
        candidate_scores: list/array of original hybrid scores
        top_k: number of final items to return
        """
        if len(candidate_items) == 0:
            return []

        candidate_items = np.array(candidate_items)
        candidate_scores = np.array(candidate_scores)

        # Precompute similarity between candidates
        cand_vectors = self.tfidf_matrix[candidate_items]
        sim_matrix = cosine_similarity(cand_vectors, cand_vectors)

        selected_indices = []
        unselected_indices = list(range(len(candidate_items)))

        # Select first item strictly by score
        first_idx = np.argmax(candidate_scores)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)

        while len(selected_indices) < top_k and len(unselected_indices) > 0:
            mmr_scores = []

            for i in unselected_indices:
                score1 = self.lambda_param * candidate_scores[i]

                # Penalty based on max similarity to already selected items
                sims = sim_matrix[i, selected_indices]
                penalty = (1.0 - self.lambda_param) * np.max(sims)

                mmr_scores.append(score1 - penalty)

            best_idx_in_unselected = np.argmax(mmr_scores)
            best_idx = unselected_indices[best_idx_in_unselected]

            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

        return candidate_items[selected_indices]
