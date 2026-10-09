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
from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_SEED, NUM_NEGATIVE_CANDIDATES, RATING_THRESHOLD

def write_report_block(f, model_name, slice_name, metrics, user_list, item_slice, evaluator, train_df, test_df):
    f.write(f"--- {slice_name.upper()} ---\n")
    f.write(f"Model: {model_name}\n")
    
    # Calculate eligible users
    eligible = 0
    relevant_items_count = 0
    for u in user_list:
        if u in evaluator.user_true_items:
            rel = evaluator.user_true_items[u]
            if item_slice is not None:
                rel = rel.intersection(item_slice)
            if len(rel) > 0:
                eligible += 1
                relevant_items_count += len(rel)
    
    f.write(f"Eligible Users: {eligible}\n")
    if eligible > 0:
        f.write(f"Avg Relevant Items per User: {relevant_items_count/eligible:.2f}\n")
    f.write(f"Candidate Settings: {NUM_NEGATIVE_CANDIDATES} sampled negatives (Seed: {RANDOM_SEED})\n")
    f.write(f"Relevance Threshold: Rating >= {RATING_THRESHOLD}\n")
    
    if metrics:
        for m, val in metrics.items():
            f.write(f"{m}: {val:.4f}\n")
    else:
        f.write("No eligible users for this slice.\n")
    f.write("\n")

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
    eval_cf = Evaluator(cf_model, train_df, test_df, num_items)
    eval_cb = Evaluator(cb_model, train_df, test_df, num_items)

    all_users = test_df["user_idx"].unique()
    sample_users = all_users

    warm_users = set(user_counts.keys()) - preprocessor.cold_users
    warm_sample = [u for u in sample_users if u in warm_users]
    cold_sample = [u for u in sample_users if u in preprocessor.cold_users]

    item_counts = train_df["item_idx"].value_counts().to_dict()
    cold_items = set([i for i in range(num_items) if item_counts.get(i, 0) < 5])
    warm_items = set([i for i in range(num_items) if item_counts.get(i, 0) >= 5])

    results = []

    print("\nEvaluating Models (This may take a minute)...")
    results.append(("All Users", "Hybrid", eval_hybrid.evaluate_users(sample_users), sample_users, None))
    results.append(("Warm Users", "Hybrid", eval_hybrid.evaluate_slice(warm_sample), warm_sample, None))
    results.append(("Cold-Start Users", "Hybrid", eval_hybrid.evaluate_slice(cold_sample), cold_sample, None))
    results.append(("Warm Items", "Hybrid", eval_hybrid.evaluate_slice(sample_users, item_slice=warm_items), sample_users, warm_items))
    results.append(("Cold Items", "Hybrid", eval_hybrid.evaluate_slice(sample_users, item_slice=cold_items), sample_users, cold_items))
    
    results.append(("Popularity Baseline", "Popularity Baseline", eval_pop.evaluate_users(sample_users), sample_users, None))
    results.append(("MMR Reranker (Hybrid)", "Hybrid + MMR", eval_mmr.evaluate_users(sample_users), sample_users, None))
    
    # Ablation Study
    results.append(("Ablation: CF Only (All Users)", "Collaborative Filtering", eval_cf.evaluate_users(sample_users), sample_users, None))
    results.append(("Ablation: CB Only (All Users)", "Content-Based Filtering", eval_cb.evaluate_users(sample_users), sample_users, None))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_file = os.path.join(REPORTS_DIR, "metrics.txt")
    print(f"\nWriting organized metrics to {metrics_file} ...")
    with open(metrics_file, "w") as f:
        for slice_name, model_name, metrics, users, item_slc in results:
            print(f"--- {slice_name} ({model_name}) ---")
            if metrics:
                for m, val in metrics.items():
                    print(f"{m}: {val:.4f}")
            
            write_report_block(f, model_name, slice_name, metrics, users, item_slc, eval_hybrid, train_df, test_df)

    print("\nGenerating Failure Analysis Report...")
    analyzer = FailureAnalyzer(hybrid_model, train_df, test_df, movies_df, eval_hybrid)
    analyzer.analyze(sample_users, os.path.join(REPORTS_DIR, "failure_analysis_details.md"))

if __name__ == "__main__":
    main()
