import pandas as pd
from src.preprocessing import Preprocessor


def test_preprocessing_temporal_split():
    df = pd.DataFrame(
        {
            "user_id": [1] * 10,
            "movie_id": list(range(10)),
            "rating": [5] * 10,
            "timestamp": list(range(10)),
        }
    )

    preprocessor = Preprocessor(cold_user_fraction=0.0)
    train_df, test_df = preprocessor.process(df)

    # 80% of 10 is 8 for train, 2 for test
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
