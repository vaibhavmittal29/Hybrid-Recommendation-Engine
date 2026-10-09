import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import TFIDF_MAX_FEATURES


class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_matrix = None
        self.user_profiles = None
        self.train_df = None

    def fit(self, train_df, movies_df, num_users, num_items):
        self.train_df = train_df

        # Combine text features
        movies_df["genres_str"] = movies_df["genres"].str.replace("|", " ")
        movies_df["content_text"] = (
            movies_df["title"]
            + " "
            + movies_df["year"].astype(str)
            + " "
            + movies_df["genres_str"]
            + " "
            + movies_df["tags"]
        )

        item_text = [""] * num_items
        for _, row in movies_df.iterrows():
            item_text[int(row["item_idx"])] = row["content_text"]

        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=TFIDF_MAX_FEATURES
        )
        self.tfidf_matrix = vectorizer.fit_transform(item_text)

        # Precompute user profiles as a weighted sum of TFIDF vectors
        self.user_profiles = np.zeros((num_users, self.tfidf_matrix.shape[1]))

        # We compute weighted profile based on explicit rating (or just average for those >= 3)
        # The prompt says: sum(r_ui * cosine) / sum(|r_ui|).
        # This implies we can just compute cosine against all user items, then weight average.
        # But for speed, predicting batch items this way:
        # User profile vector = sum(r_ui * v_i) / sum(r_ui) ? No, cosine is non-linear.
        # If we just do: Score = sum(r * cos) / sum(r). We can do that
        # dynamically in predict.

        # To optimize, we store the historical positive items and ratings per user.
        self.user_history = {}
        for u, group in train_df.groupby("user_idx"):
            # Filter to positively rated items only
            pos_mask = group["rating"] >= 3.0
            items = group.loc[pos_mask, "item_idx"].values
            ratings = group.loc[pos_mask, "rating"].values
            self.user_history[int(u)] = (items, ratings)

    def predict_batch_users(self, user_idx, all_items):
        if user_idx not in self.user_history:
            return np.zeros(len(all_items))

        hist_items, hist_ratings = self.user_history[user_idx]

        if len(hist_items) == 0:
            # If user has no positive ratings, content-based score is 0
            return np.zeros(len(all_items))

        hist_vectors = self.tfidf_matrix[hist_items]
        target_vectors = self.tfidf_matrix[all_items]

        # shape: (len(hist_items), len(all_items))
        sim_matrix = cosine_similarity(hist_vectors, target_vectors)

        # Weight by ratings
        # sim_matrix is (num_hist, num_targets)
        # hist_ratings is (num_hist,)
        weighted_sim = sim_matrix.T.dot(hist_ratings) / (hist_ratings.sum() + 1e-9)

        return weighted_sim
