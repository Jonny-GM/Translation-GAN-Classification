import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from pathlib import Path


def dats_to_torch(f, type_ds):
    lines = open(f, "r").readlines()
    skip = next(i for i, v in enumerate(lines) if not v.startswith("@"))
    df = pd.read_csv(f, header=None, skiprows=skip)
    df = df.rename({df.shape[1] - 1: "class"}, axis=1)
    df = df.replace({" negative": 0, " positive": 1})
    df = df.replace({"M": 0, "F": 1, "I": 2})
    X = StandardScaler(with_std=True).fit_transform(
        df[[c for c in df.columns if c != "class"]]
    )
    y = df["class"]
    torch.save(
        [
            torch.from_numpy(X),
            torch.from_numpy(y),
        ],
        (f.parent / "ds_{}.pt".format(type_ds)),
    )


for ds_path in Path("datasets_keel").iterdir():
    for f in ds_path.iterdir():
        if "dat" in f.suffix:
            if "1tra" in f.name:
                dats_to_torch(f, "train")
            if "1tst" in f.name:
                dats_to_torch(f, "test")
