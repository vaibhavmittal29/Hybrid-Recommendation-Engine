import pytest
import numpy as np
import pandas as pd
from src.preprocessing import Preprocessor
from src.evaluation import Evaluator, precision_at_k, recall_at_k, ndcg_at_k
from src.content_based import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender
from src.failure_analysis import FailureAnalyzer
from src.baseline import PopularityBaseline

def test_item_classification_counts():
    train_df = pd.DataFrame({
        "user_idx": [1, 2, 3, 4, 1, 2, 3, 4, 5],
        "item_idx": [0, 0, 0, 0, 1, 1, 1, 1, 1],
        "rating": [5] * 9
    })
    
    item_counts = train_df["item_idx"].value_counts().to_dict()
    cold_items = set([i for i in range(3) if item_counts.get(i, 0) < 5])
    warm_items = set([i for i in range(3) if item_counts.get(i, 0) >= 5])
    
    assert 0 in cold_items
    assert 2 in cold_items
    assert 1 in warm_items
    assert 1 not in cold_items
    assert 0 not in warm_items

def test_missing_metadata_empty_profiles():
    movies_df = pd.DataFrame({
        "item_idx": [0, 1],
        "movie_id": [1, 2],
        "title": ["A", ""],
        "genres": ["Action", None],
        "tags": [None, ""],
        "year": [None, "2020"]
    })
    movies_df["genres_str"] = movies_df["genres"].fillna("").str.replace("|", " ")
    movies_df["tags"] = movies_df["tags"].fillna("")
    movies_df["content_text"] = movies_df["title"].fillna("") + " " + movies_df["genres_str"]
    
    cb = ContentBasedRecommender()
    cb.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0], "rating": [5.0]}), movies_df, 1, 2)
    
    assert cb.tfidf_matrix.shape[0] == 2

def test_ranking_metrics_edge_cases():
    assert recall_at_k([1, 2, 3], [], 3) == 0.0
    assert ndcg_at_k([1, 2, 3], [], 3) == 0.0
    assert precision_at_k([], [1, 2], 5) == 0.0
    assert precision_at_k([1], [1], 5) == 1.0 / 5.0

def test_score_normalization_constant_array():
    class DummyCF:
        def predict_batch_users(self, u, items):
            return np.ones(len(items))
            
    hybrid = HybridRecommender(DummyCF(), DummyCF(), user_counts={0: 100})
    scores, _, _, _ = hybrid.predict_batch_users(0, np.array([0, 1, 2]))
    assert np.allclose(scores, 0.0)

def test_baseline_comparisons_toy_data():
    train_df = pd.DataFrame({
        "user_idx": [0, 0, 1, 1],
        "item_idx": [0, 0, 1, 0],
        "rating": [5, 5, 5, 5]
    })
    pop_model = PopularityBaseline(train_df, 3)
    scores = pop_model.predict_batch_users(99, np.array([0, 1, 2]))
    if isinstance(scores, tuple):
        scores = scores[0]
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]
