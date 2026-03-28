import os
import pandas as pd
from torch.utils.data import Dataset

_DATA_DIR = os.path.dirname(__file__)

class IMDBDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data_frame = pd.read_csv(os.path.join(_DATA_DIR, "imdb", f"{self.split}.csv"), names=["label", "text"], dtype={"label": int, "text": str})

        print(self.data_frame.head())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return row["text"], row["label"]

class YelpDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data_frame = pd.read_csv(os.path.join(_DATA_DIR, "yelp", f"{self.split}.csv"), names=["label", "text"], dtype={"label": str, "text": str})
        self.data_frame["label"] = self.data_frame["label"].str.strip('"').astype(int)
        self.data_frame["label"] -= 1 # align to 0-4

        print(self.data_frame.head())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return row["text"], row["label"]

class AmazonDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data_frame = pd.read_csv(os.path.join(_DATA_DIR, "amazon", f"{self.split}.csv"), names=["label", "title", "review"], dtype={"label": str, "title": str, "review": str})
        self.data_frame["text"] = self.data_frame["title"].fillna("") + " " + self.data_frame["review"].fillna("")
        self.data_frame["label"] = self.data_frame["label"].str.strip('"')
        self.data_frame = self.data_frame.dropna(subset=["label"])
        self.data_frame = self.data_frame[self.data_frame["label"] != ""]
        self.data_frame["label"] = self.data_frame["label"].astype(int)
        self.data_frame["label"] -= 1 # align to 0-4
        self.data_frame.drop(columns=["title", "review"], inplace=True)

        print(self.data_frame.head())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return row["text"], row["label"]


# Smoke test
if __name__ == "__main__":
    d = YelpDataset()

    print("=" * 40)

    d = AmazonDataset()
