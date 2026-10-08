import pandas as pd
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
