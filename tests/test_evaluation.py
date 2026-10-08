import numpy as np
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k


def test_metrics():
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 5, 6]

    # Hits at 5: 2 and 5 (2 hits)
    # Precision@5 = 2 / 5 = 0.4
    assert np.isclose(precision_at_k(recommended, relevant, 5), 0.4)

    # Recall@5 = 2 / 3 = 0.666...
    assert np.isclose(recall_at_k(recommended, relevant, 5), 2.0 / 3.0)

    # NDCG@5
    # rel at pos 2 (index 1), pos 5 (index 4)
    dcg = (1 / np.log2(1 + 2)) + (1 / np.log2(4 + 2))
    # idcg = (1/np.log2(0+2)) + (1/np.log2(1+2)) + (1/np.log2(2+2))
    idcg = (1 / 1.0) + (1 / np.log2(3)) + (1 / np.log2(4))
    assert np.isclose(ndcg_at_k(recommended, relevant, 5), dcg / idcg)
