import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.reranker import MMRReranker
from src.hybrid_recommender import HybridRecommender
from src.config import MODELS_DIR
warnings.filterwarnings("ignore")


def main(user_id):
    try:
        with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "rb") as f:
            preprocessor = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "cf_model.pkl"), "rb") as f:
            cf_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "cb_model.pkl"), "rb") as f:
            cb_model = pickle.load(f)

        train_df = pd.read_pickle(os.path.join(MODELS_DIR, "train_df.pkl"))
        movies_df = pd.read_pickle(os.path.join(MODELS_DIR, "movies_df.pkl"))
    except FileNotFoundError:
        print("Models not found. Run 'python scripts/train.py' first.")
        return

    if user_id not in preprocessor.user_map:
        print(f"User {user_id} not found in training data.")
        return

    user_idx = preprocessor.user_map[user_id]
    num_items = len(preprocessor.item_map)

    user_counts = train_df["user_idx"].value_counts().to_dict()
    hybrid_model = HybridRecommender(cf_model, cb_model, user_counts)
    reranker = MMRReranker(cb_model.tfidf_matrix)

    all_items = np.arange(num_items)
    scores, cf_scores, cb_scores, alpha = hybrid_model.predict_batch_users(
        user_idx, all_items
    )

    train_items = train_df[train_df["user_idx"] == user_idx]["item_idx"].tolist()
    scores[train_items] = -np.inf

    # Rerank Top 50 using MMR to get Top 10
    pool_indices = np.argsort(scores)[::-1][:50]
    pool_items = all_items[pool_indices]
    pool_scores = scores[pool_indices]

    top_items = reranker.rerank(user_idx, pool_items, pool_scores, top_k=10)

    print(f"Top 10 Recommendations for User {user_id} (History Length: {
            user_counts.get(
                user_idx,
                0)}):")
    print(f"Blend Weight: {alpha:.2f} (CF) / {1.0 - alpha:.2f} (CB)")
    print("-" * 50)

    for rank, idx in enumerate(top_items):
        preprocessor.reverse_item_map[idx]
        movie_row = movies_df[movies_df["item_idx"] == idx].iloc[0]

        # We need the index in original scores array
        original_idx = np.where(all_items == idx)[0][0]
        f_score = scores[original_idx]

        print(f"{
                rank +
                1}. {
                movie_row['title']} (Year: {
                movie_row['year']}, Genres: {
                    movie_row['genres']})")
        print(f"   Hybrid Score: {
                f_score:.4f} | CF: {
                cf_scores[original_idx]:.4f} | CB: {
                cb_scores[original_idx]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/recommend.py <user_id>")
    else:
        main(int(sys.argv[1]))
