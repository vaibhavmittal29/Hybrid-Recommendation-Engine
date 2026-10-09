# Hybrid Recommendation Engine with Cold-Start Handling

## Scenario
A streaming platform wants to recommend content to users, but new users and new items have no interaction history — a naive collaborative filter fails for both. 

## Objective
To build a hybrid recommender that blends collaborative filtering (CF) with content-based (CB) features so it degrades gracefully for cold-start cases instead of breaking. The blending strategy dynamically shifts weight towards CB scores as interaction history becomes sparse, making it effective for cold-start scenarios.

## Dataset
- **MovieLens 1M Dataset**
- **Source**: GroupLens / Kaggle
- **Fields**: 
  - `ratings.dat`: UserID::MovieID::Rating::Timestamp
  - `movies.dat`: MovieID::Title::Genres

### Cold-Start Simulation & Evaluation
- **Cold Users**: The dataset is processed to artificially create a "Cold-Start Slice" for users. 10% of users are randomly selected, and their training interactions are aggressively truncated to simulate a cold-start state (`< 5` interactions).
- **Cold Items**: Item interaction counts are computed strictly using the training split (preventing train/test leakage). Items with fewer than 5 training interactions are evaluated separately as **Cold Items**, ensuring the system can recommend sparse or unseen catalog items robustly using their content metadata.

## Approach

### Collaborative Filtering (CF)
We employ Matrix Factorization using **Surprise SVD** (Singular Value Decomposition).
- The model is built utilizing the `scikit-surprise` library, which inherently handles user and item latent factors.
- Base predictions are generated internally by `Surprise`, yielding robust collaborative scores that excel with dense interaction history.

### Content-Based Recommendation (CB)
We employ Term Frequency - Inverse Document Frequency (TF-IDF) across movie `genres` and `title` fields.
- Items are represented as numerical vectors in the TF-IDF feature space.
- A user's profile vector is built by aggregating the TF-IDF vectors of items they have rated positively (rating >= 3.0).
- Predictions are made by calculating the `cosine similarity` between the user profile and candidate item profiles. Unseen or sparse items naturally receive meaningful content-based scoring if their metadata is available.

### Adaptive Blending
We use dynamic blending via a **Sigmoid weighting mechanism** to gracefully degrade to content features.
- Scores from the CF and CB models are first Min-Max normalized to `[0, 1]` independently.
- The collaborative-filtering weight ($\alpha$) is derived using a logistic sigmoid function based on the user's interaction count ($n_u$):
  $$ \alpha = \frac{1}{1 + e^{-\beta (n_u - \tau)}} $$
- **CF Weight** = $\alpha$
- **CB Weight** = $1.0 - \alpha$
- For sparse/cold-start users (e.g., $n_u < 5$), $\alpha$ approaches 0, severely diminishing the CF Weight and allowing the Content-Based (CB) Weight to dominate. Conversely, for users with abundant interactions, $\alpha$ approaches 1, prioritizing the CF model.

### Evaluation Metrics
We use standard ranking metrics (`Precision@K`, `Recall@K`, `NDCG@K`) separated into multiple distinct evaluation slices:
- All Users
- Warm Users / Cold-Start Users
- Warm Items / Cold Items
- Popularity Baseline / MMR Reranker

## Setup & Dependency Installation

1. **Clone the repository**
```bash
git clone https://github.com/vaibhavmittal29/Hybrid-Recommendation-Engine.git
cd Hybrid-Recommendation-Engine
```

2. **Setup Virtual Environment & Install Dependencies**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# On Unix / macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Download Dataset**
```bash
# Works identically across OS from repository root
python scripts/download_data.py
```

## How to Run & Reproduce

**1. Running Tests**
Run the full pytest suite to validate system internals (cold/warm categorization, positive rating filters, adaptive weighting, slicing):
```bash
pytest tests/
```

**2. Training the Models**
This will pre-process the data, simulate cold users, build the CF and CB models, and save them in the `models/` directory.
```bash
python scripts/train.py
```

**3. Evaluate the System**
This runs the full evaluation pipeline, splitting metrics among user and item slices. It evaluates the hybrid model consistently alongside a popularity baseline, utilizing a pinned random seed for rigorous reproducibility. It also generates a detailed `failure_analysis_details.md`.
```bash
python scripts/evaluate.py
```

**4. Generate Recommendations**
To generate real-time recommendations for a specific user:
```bash
# Example: Generating recommendations for user ID 1
python scripts/recommend.py 1
```

## Expected Results & Failure Analysis
You will find detailed evaluation output in `reports/metrics.txt` after running the evaluate script. The hybrid model maintains robustness across both cold-start users and cold items. 
The system automatically generates a deterministic Failure Analysis report in the `reports/` folder. It scientifically categorizes complete recommendation misses (`NDCG@10 = 0.0`) by directly inspecting TF-IDF structural sparsity (for metadata poverty) and baseline ranking distributions (for popularity bias) rather than blindly assigning generic failure buckets.
