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
- The dataset is processed to artificially create a "Cold-Start Slice" for users and items with `< 5` interactions to explicitly test the adaptive behavior of the hybrid engine. 

## Approach

### Collaborative Filtering (CF)
We employ Matrix Factorization using **Surprise SVD** (Singular Value Decomposition).
- The model is built utilizing the `scikit-surprise` library, which inherently handles user and item latent factors.
- Base predictions are generated internally by `Surprise`, yielding robust collaborative scores that excel with dense interaction history.

### Content-Based Recommendation (CB)
We employ Term Frequency - Inverse Document Frequency (TF-IDF) across movie `genres` and `title` fields.
- Items are represented as numerical vectors in the TF-IDF feature space.
- A user's profile vector is built by aggregating the TF-IDF vectors of items they have rated positively (rating >= 3.0).
- Predictions are made by calculating the `cosine similarity` between the user profile and candidate item profiles.

### Adaptive Blending
We use dynamic blending via a **Sigmoid weighting mechanism** to gracefully degrade to content features.
- Scores from the CF and CB models are first Min-Max normalized to `[0, 1]` independently.
- The collaborative-filtering weight ($\alpha$) is derived using a logistic sigmoid function based on the user's interaction count ($n_u$):
  $$ \alpha = \frac{1}{1 + e^{-\beta (n_u - \tau)}} $$
- **CF Weight** = $\alpha$
- **CB Weight** = $1.0 - \alpha$
- The final hybrid score is the weighted sum of both normalized scores. 
- For sparse/cold-start users (e.g., $n_u < 5$), $\alpha$ approaches 0, severely diminishing the CF Weight and allowing the Content-Based (CB) Weight to dominate. Conversely, for users with abundant interactions, $\alpha$ approaches 1, prioritizing the CF model. This mathematical formulation allows the system to smoothly and dynamically handle cold-start situations without hard-coded cutoffs.

### Evaluation
We use standard ranking metrics (`Precision@K`, `Recall@K`, `NDCG@K`) applied to:
- **Warm Users**: Users with `>= 5` interactions in training.
- **Cold Users**: Users with `< 5` interactions in training (explicitly simulated during pre-processing).

## Setup & Dependency Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/hybrid-recommendation-engine.git
cd hybrid-recommendation-engine
```

2. **Setup Virtual Environment & Install Dependencies**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

pip install -r requirements.txt
```

3. **Download Dataset**
```bash
python scripts/download_data.py
```

## How to Run & Reproduce

**1. Training the Models**
This will pre-process the data, build the CF and CB models, and save them in the `models/` directory.
```bash
python scripts/train.py
```

**2. Evaluate the System**
This runs the full evaluation pipeline, splitting metrics among all users, warm users, and cold-start users.
```bash
python scripts/evaluate.py
```

**3. Generate Recommendations for a specific user**
```bash
# E.g. generating recommendations for user ID 1
python scripts/recommend.py 1
```

## Expected Results
You will find detailed evaluation output in `reports/metrics.txt` after running the evaluate script. The hybrid model exhibits relatively stable metrics across the cold-start slice by leveraging the fallback to the CB model.

## Failure Analysis
In cold-start scenarios with poor metadata (e.g., highly generic genres like "Drama"), the CB component struggles to differentiate items effectively. For very sparse users where there are almost zero interactions, popularity bias may take over or the model may recommend loosely matched items due to broad cosine similarities.
