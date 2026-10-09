import numpy as np
import pandas as pd
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k, Evaluator

def test_metrics():
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 5, 6]

    assert np.isclose(precision_at_k(recommended, relevant, 5), 0.4)
    assert np.isclose(recall_at_k(recommended, relevant, 5), 2.0 / 3.0)

    dcg = (1 / np.log2(1 + 2)) + (1 / np.log2(4 + 2))
    idcg = (1 / 1.0) + (1 / np.log2(3)) + (1 / np.log2(4))
    assert np.isclose(ndcg_at_k(recommended, relevant, 5), dcg / idcg)

class DummyRecommender:
    def predict_batch_users(self, u, candidates):
        # Just return scores equal to item indices for deterministic sorting
        # Recommends higher index items first
        return candidates.astype(float), None, None, None

def test_evaluator_item_slice():
    train_df = pd.DataFrame({"user_idx": [0], "item_idx": [0], "rating": [5.0]})
    # User 0 likes items 1 and 2 in test set
    test_df = pd.DataFrame({
        "user_idx": [0, 0], 
        "item_idx": [1, 2], 
        "rating": [5.0, 5.0]
    })

    evaluator = Evaluator(DummyRecommender(), train_df, test_df, num_items=5, k_list=[2])
    
    # Evaluate with slice containing only item 1 and item 4 (negative)
    # The relevant item is 1. The negatives will be drawn from all available.
    # We restrict available negatives to item_slice. Available globally: {3, 4} (since 0, 1, 2 are in history/test)
    # Available negatives in item_slice {1, 4} that are not in history is just {4}.
    item_slice = {1, 4}
    
    # 1. Explicitly verify that negative sampling respects the slice
    negatives = evaluator._sample_negatives(0, num_negatives=5, item_slice=item_slice)
    assert len(negatives) > 0, "Should have sampled at least one negative"
    for neg in negatives:
        assert neg in item_slice, f"Sampled negative {neg} is not in the requested item_slice"
        assert neg != 0, "Sampled negative should not be in training history"

    # 2. Verify end-to-end evaluation runs successfully
    metrics = evaluator.evaluate_slice([0], item_slice=item_slice)
    assert metrics is not None
    assert "Precision@2" in metrics
