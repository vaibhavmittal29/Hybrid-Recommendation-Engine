from surprise import Dataset, Reader, SVD
import numpy as np
from src.config import SVD_N_FACTORS, SVD_N_EPOCHS, SVD_LR_ALL, SVD_REG_ALL, RANDOM_SEED


class CollaborativeFilteringSVD:
    def __init__(self):
        self.model = SVD(
            n_factors=SVD_N_FACTORS,
            n_epochs=SVD_N_EPOCHS,
            lr_all=SVD_LR_ALL,
            reg_all=SVD_REG_ALL,
            random_state=RANDOM_SEED,
        )
        self.global_mean = 0

    def fit(self, train_df, num_users, num_items):
        self.global_mean = train_df["rating"].mean()

        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["user_idx", "item_idx", "rating"]], reader
        )

        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        self.trainset = trainset

    def predict_batch_users(self, user_idx, all_items):
        # We need to predict ratings for a single user over all items
        # Surprise prediction returns an Estimate.
        preds = []
        for i in all_items:
            # uid and iid are raw ids to the model if we passed raw ids, which we did (user_idx, item_idx)
            # Actually trainset knows 'user_idx' as raw id.
            pred = self.model.predict(uid=user_idx, iid=i)
            preds.append(pred.est)
        return np.array(preds)
