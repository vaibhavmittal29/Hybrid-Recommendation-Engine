import pandas as pd
import numpy as np
from src.content_based import ContentBasedRecommender


def test_cb_fit_predict():
    train_df = pd.DataFrame({"user_idx": [0], "item_idx": [0], "rating": [5.0]})

    movies_df = pd.DataFrame(
        {
            "item_idx": [0, 1],
            "title": ["Movie A", "Movie B"],
            "year": ["2000", "2001"],
            "genres": ["Action", "Action|Comedy"],
            "tags": ["", ""],
        }
    )

    cb = ContentBasedRecommender()
    cb.fit(train_df, movies_df, num_users=1, num_items=2)

    preds = cb.predict_batch_users(0, [0, 1])
    assert len(preds) == 2
    assert preds[0] > 0.0  # Should be somewhat similar to itself

def test_cb_positive_rating_filter_and_sparse():
    # User 0 only has a negative rating (2.0)
    # User 1 has no history (not in train_df)
    train_df = pd.DataFrame({"user_idx": [0], "item_idx": [0], "rating": [2.0]})

    movies_df = pd.DataFrame(
        {
            "item_idx": [0, 1],
            "title": ["Movie A", "Movie B"],
            "year": ["2000", "2001"],
            "genres": ["Action", "Action|Comedy"],
            "tags": ["", ""],
        }
    )

    cb = ContentBasedRecommender()
    cb.fit(train_df, movies_df, num_users=2, num_items=2)

    # User 0 should have 0 score because rating < 3.0
    preds_0 = cb.predict_batch_users(0, [0, 1])
    assert np.all(preds_0 == 0.0)

    # User 1 should have 0 score because sparse (no history)
    preds_1 = cb.predict_batch_users(1, [0, 1])
    assert np.all(preds_1 == 0.0)
