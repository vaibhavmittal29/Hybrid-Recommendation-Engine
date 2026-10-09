import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import warnings
warnings.filterwarnings("ignore")

from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringSVD
from src.preprocessing import Preprocessor
from src.data_loader import DataLoader
from src.config import MODELS_DIR
warnings.filterwarnings("ignore")


def main():
    print("Loading data...")
    loader = DataLoader()
    ratings = loader.load_ratings()
    movies = loader.load_movies()

    print("Preprocessing data...")
    preprocessor = Preprocessor()
    train_df, test_df = preprocessor.process(ratings)
    movies_df = preprocessor.process_movies(movies)

    num_users = len(preprocessor.user_map)
    num_items = len(preprocessor.item_map)

    print(f"Total Users: {num_users}, Total Items: {num_items}")

    print("Training Collaborative Filtering model (SVD)...")
    cf_model = CollaborativeFilteringSVD()
    cf_model.fit(train_df, num_users, num_items)

    print("Training Content-Based model (TF-IDF)...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(train_df, movies_df, num_users, num_items)

    print("Saving models and data splits...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(MODELS_DIR, "cf_model.pkl"), "wb") as f:
        pickle.dump(cf_model, f)

    with open(os.path.join(MODELS_DIR, "cb_model.pkl"), "wb") as f:
        pickle.dump(cb_model, f)

    train_df.to_pickle(os.path.join(MODELS_DIR, "train_df.pkl"))
    test_df.to_pickle(os.path.join(MODELS_DIR, "test_df.pkl"))
    movies_df.to_pickle(os.path.join(MODELS_DIR, "movies_df.pkl"))

    print("Training complete. Models saved to models/ directory.")


if __name__ == "__main__":
    main()
