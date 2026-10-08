import os
import urllib.request
import zipfile


def download_movielens_1m(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extract_dir = os.path.join(data_dir, "ml-1m")

    if not os.path.exists(extract_dir):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Done!")
    else:
        print("MovieLens 1M dataset already exists.")


if __name__ == "__main__":
    download_movielens_1m()
