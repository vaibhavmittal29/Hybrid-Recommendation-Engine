import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.failure_analysis import FailureAnalyzer
from src.reranker import MMRReranker
from src.baseline import PopularityBaseline
from src.evaluation import Evaluator
from src.hybrid_recommender import HybridRecommender
from src.config import MODELS_DIR, REPORTS_DIR
warnings.filterwarnings("ignore")


def main():
    print("Loading models and data...")
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "cf_model.pkl"), "rb") as f:
        cf_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "cb_model.pkl"), "rb") as f:
        cb_model = pickle.load(f)

    train_df = pd.read_pickle(os.path.join(MODELS_DIR, "train_df.pkl"))
    test_df = pd.read_pickle(os.path.join(MODELS_DIR, "test_df.pkl"))
    movies_df = pd.read_pickle(os.path.join(MODELS_DIR, "movies_df.pkl"))

    num_items = len(preprocessor.item_map)
    user_counts = train_df["user_idx"].value_counts().to_dict()

    print("Initializing Models...")
    hybrid_model = HybridRecommender(cf_model, cb_model, user_counts)
    pop_model = PopularityBaseline(train_df, num_items)
    reranker = MMRReranker(cb_model.tfidf_matrix)

    eval_hybrid = Evaluator(hybrid_model, train_df, test_df, num_items)
    eval_pop = Evaluator(pop_model, train_df, test_df, num_items)
    eval_mmr = Evaluator(hybrid_model, train_df, test_df, num_items, reranker=reranker)

    all_users = test_df["user_idx"].unique()

    # Sub-sample for faster eval if needed, but the prompt says:
    # "Evaluate across all test users using 500 sampled negative candidates"
    # With 500 negatives, eval is fast enough for all users.
    sample_users = all_users

    print("\n--- 1. Evaluating ALL Users (Hybrid) ---")
    all_metrics = eval_hybrid.evaluate_users(sample_users)
    for m, val in all_metrics.items():
        print(f"{m}: {val:.4f}")

    print("\n--- 2. Evaluating WARM Users (Hybrid) ---")
    warm_users = set(user_counts.keys()) - preprocessor.cold_users
    warm_sample = [u for u in sample_users if u in warm_users]
    warm_metrics = eval_hybrid.evaluate_slice(warm_sample)
    if warm_metrics:
        for m, val in warm_metrics.items():
            print(f"{m}: {val:.4f}")

    print("\n--- 3. Evaluating COLD-START Users (Hybrid) ---")
    cold_sample = [u for u in sample_users if u in preprocessor.cold_users]
    cold_metrics = eval_hybrid.evaluate_slice(cold_sample)
    if cold_metrics:
        for m, val in cold_metrics.items():
            print(f"{m}: {val:.4f}")
            
    # Calculate item counts using training split only
    item_counts = train_df["item_idx"].value_counts().to_dict()
    cold_items = set([i for i in range(num_items) if item_counts.get(i, 0) < 5])
    warm_items = set([i for i in range(num_items) if item_counts.get(i, 0) >= 5])
    
    print("\n--- 4. Evaluating COLD ITEMS (Hybrid) ---")
    cold_item_metrics = eval_hybrid.evaluate_slice(sample_users, item_slice=cold_items)
    if cold_item_metrics:
        for m, val in cold_item_metrics.items():
            print(f"{m}: {val:.4f}")

    print("\n--- 5. Evaluating WARM ITEMS (Hybrid) ---")
    warm_item_metrics = eval_hybrid.evaluate_slice(sample_users, item_slice=warm_items)
    if warm_item_metrics:
        for m, val in warm_item_metrics.items():
            print(f"{m}: {val:.4f}")

    print("\n--- 6. Evaluating Popularity Baseline (Bonus) ---")
    pop_metrics = eval_pop.evaluate_users(sample_users)
    for m, val in pop_metrics.items():
        print(f"{m}: {val:.4f}")

    print("\n--- 7. Evaluating MMR Reranker (Bonus) ---")
    mmr_metrics = eval_mmr.evaluate_users(sample_users)
    for m, val in mmr_metrics.items():
        print(f"{m}: {val:.4f}")

    print("\nGenerating Failure Analysis Report...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    analyzer = FailureAnalyzer(hybrid_model, train_df, test_df, movies_df, eval_hybrid)
    analyzer.analyze(
        sample_users, os.path.join(REPORTS_DIR, "failure_analysis_details.md")
    )

    with open(os.path.join(REPORTS_DIR, "metrics.txt"), "w") as f:
        f.write("--- ALL USERS ---\n")
        for m, val in all_metrics.items():
            f.write(f"{m}: {val:.4f}\n")

        f.write("\n--- WARM USERS ---\n")
        if warm_metrics:
            for m, val in warm_metrics.items():
                f.write(f"{m}: {val:.4f}\n")
                
        f.write("\n--- COLD-START USERS ---\n")
        if cold_metrics:
            for m, val in cold_metrics.items():
                f.write(f"{m}: {val:.4f}\n")
                
        f.write("\n--- WARM ITEMS ---\n")
        if warm_item_metrics:
            for m, val in warm_item_metrics.items():
                f.write(f"{m}: {val:.4f}\n")
                
        f.write("\n--- COLD ITEMS ---\n")
        if cold_item_metrics:
            for m, val in cold_item_metrics.items():
                f.write(f"{m}: {val:.4f}\n")

        f.write("\n--- POPULARITY BASELINE ---\n")
        for m, val in pop_metrics.items():
            f.write(f"{m}: {val:.4f}\n")


if __name__ == "__main__":
    main()
