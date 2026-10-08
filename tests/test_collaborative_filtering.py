import pandas as pd
from src.collaborative_filtering import CollaborativeFilteringSVD


def test_cf_fit_predict():
    train_df = pd.DataFrame(
        {"user_idx": [0, 0, 1], "item_idx": [0, 1, 0], "rating": [5.0, 4.0, 3.0]}
    )

    cf = CollaborativeFilteringSVD()
    cf.model.n_epochs = 1
    cf.fit(train_df, num_users=2, num_items=2)

    preds = cf.predict_batch_users(0, [0, 1])
    assert len(preds) == 2
