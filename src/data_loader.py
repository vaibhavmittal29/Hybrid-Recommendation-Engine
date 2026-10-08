import os
import pandas as pd
from src.config import DATA_DIR, DATASET_NAME


class DataLoader:
    def __init__(self, dataset_name=DATASET_NAME):
        self.dataset_name = dataset_name
        self.data_path = os.path.join(DATA_DIR, dataset_name)

    def load_ratings(self):
        if self.dataset_name == "ml-1m":
            file_path = os.path.join(self.data_path, "ratings.dat")
            df = pd.read_csv(
                file_path,
                sep="::",
                engine="python",
                names=["user_id", "movie_id", "rating", "timestamp"],
                encoding="latin-1",
            )
        elif self.dataset_name == "ml-25m":
            file_path = os.path.join(self.data_path, "ratings.csv")
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported dataset")
        return df

    def load_movies(self):
        if self.dataset_name == "ml-1m":
            file_path = os.path.join(self.data_path, "movies.dat")
            df = pd.read_csv(
                file_path,
                sep="::",
                engine="python",
                names=["movie_id", "title", "genres"],
                encoding="latin-1",
            )
            # Extract year from title
            df["year"] = df["title"].str.extract(r"\((\d{4})\)")
            df["title"] = (
                df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
            )
            df["tags"] = ""
        elif self.dataset_name == "ml-25m":
            file_path = os.path.join(self.data_path, "movies.csv")
            df = pd.read_csv(file_path)
            df["year"] = df["title"].str.extract(r"\((\d{4})\)")
            df["title"] = (
                df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
            )

            tags_path = os.path.join(self.data_path, "tags.csv")
            if os.path.exists(tags_path):
                tags = pd.read_csv(tags_path)
                tags["tag"] = tags["tag"].fillna("")
                grouped_tags = (
                    tags.groupby("movieId")["tag"]
                    .apply(lambda x: " ".join(x))
                    .reset_index()
                )
                grouped_tags.rename(columns={"movieId": "movie_id"}, inplace=True)
                df.rename(columns={"movieId": "movie_id"}, inplace=True)
                df = df.merge(grouped_tags, on="movie_id", how="left")
                df["tag"] = df["tag"].fillna("")
                df.rename(columns={"tag": "tags"}, inplace=True)
            else:
                df.rename(columns={"movieId": "movie_id"}, inplace=True)
                df["tags"] = ""
        else:
            raise ValueError("Unsupported dataset")

        df["year"] = df["year"].fillna("0000")
        return df
