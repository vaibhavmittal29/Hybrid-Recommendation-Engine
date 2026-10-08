import numpy as np
import pandas as pd
from src.config import COLD_START_THRESHOLD, RANDOM_SEED


class Preprocessor:
    def __init__(self, cold_user_fraction=0.1, cold_threshold=COLD_START_THRESHOLD):
        self.cold_user_fraction = cold_user_fraction
        self.cold_threshold = cold_threshold
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.cold_users = set()

    def process(self, ratings):
        np.random.seed(RANDOM_SEED)

        # 1. Map IDs to contiguous indices
        unique_users = ratings["user_id"].unique()
        unique_items = ratings["movie_id"].unique()

        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {item: i for i, item in enumerate(unique_items)}
        self.reverse_user_map = {i: u for u, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        df = ratings.copy()
        df["user_idx"] = df["user_id"].map(self.user_map)
        df["item_idx"] = df["movie_id"].map(self.item_map)

        # 2. Temporal Train-Test Split (80% train, 20% test per user)
        df.sort_values(by=["user_idx", "timestamp"], inplace=True)

        # Calculate split index per user
        df["rank"] = df.groupby("user_idx").cumcount()
        df["total_count"] = df.groupby("user_idx")["user_idx"].transform("count")

        # Train if rank < 0.8 * total_count
        is_train = df["rank"] < (df["total_count"] * 0.8)

        train_df = df[is_train].copy()
        test_df = df[~is_train].copy()

        train_df.drop(columns=["rank", "total_count"], inplace=True)
        test_df.drop(columns=["rank", "total_count"], inplace=True)

        # 3. Create cold-start slice
        # Select 10% of users to be artificially "cold"
        unique_train_users = train_df["user_idx"].unique()
        num_cold_users = int(len(unique_train_users) * self.cold_user_fraction)
        cold_user_indices = np.random.choice(
            unique_train_users, size=num_cold_users, replace=False
        )

        # For cold users, keep only the FIRST `cold_threshold - 1` interactions
        # in training
        def truncate_cold_users(df, cold_keys, col_name, threshold):
            mask = df[col_name].isin(cold_keys)
            cold_df = df[mask]

            # Since they are chronologically sorted, just take head(threshold)
            truncated_cold = cold_df.groupby(col_name, as_index=False).head(threshold)

            return pd.concat([df[~mask], truncated_cold], ignore_index=True)

        train_df = truncate_cold_users(
            train_df, cold_user_indices, "user_idx", self.cold_threshold - 1
        )

        user_counts = train_df["user_idx"].value_counts()
        self.cold_users = set(user_counts[user_counts < self.cold_threshold].index)

        return train_df, test_df

    def process_movies(self, movies):
        df = movies.copy()
        df = df[df["movie_id"].isin(self.item_map.keys())]
        df["item_idx"] = df["movie_id"].map(self.item_map)
        return df
