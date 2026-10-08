# Data Directory

This directory stores the datasets.

## MovieLens 1M
Run `python scripts/download_data.py` to automatically download and extract the dataset here.
It will create `data/ml-1m/` containing:
- `ratings.dat`
- `movies.dat`
- `users.dat`

## MovieLens 25M (Optional)
To use ML-25M, download it from GroupLens and extract it so that `data/ml-25m/` contains `ratings.csv`, `movies.csv`, and `tags.csv`. Update `DATASET_NAME` in `src/config.py` to use it.
