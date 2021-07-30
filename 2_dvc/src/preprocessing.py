#!/usr/bin/python3
"""Preprocessing script

Script for preprocessing. Call with path to csv
"""

import pandas as pd
import numpy as np
import pickle
import click
import io
import os
import yaml

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords")
params = yaml.safe_load(open("params.yaml"))["preprocessing"]


class Preprocessing:
    def __init__(self):
        pass

    def __call__(self, path_to_csv):
        self.path_to_csv = path_to_csv

        return self.preprocess(self.path_to_csv)

    def preprocess(self, path_to_csv):

        df = pd.read_csv(path_to_csv)

        russian_stopwords = stopwords.words("russian")
        docs = np.array(df["steps"])

        tokenizer = RegexpTokenizer(r"\w+")
        for idx in range(len(docs)):
            docs[idx] = tokenizer.tokenize(str(docs[idx]))

        docs = [
            [token for token in doc if not token.isdigit()] for doc in docs
        ]
        docs = [
            [token for token in doc if len(token) > 3] for doc in docs
        ]
        docs = [
            [token for token in doc if token not in russian_stopwords] for doc in docs
        ]

        return docs


@click.command()
@click.option("-p", "--path_to_csv", required=True, type=str)
def main(path_to_csv):
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
    output_path = os.path.join("data", "prepared", "docs.pkl")
    preprocesser = Preprocessing()
    docs = preprocesser(path_to_csv)
    with io.open(output_path, "wb") as out:
        pickle.dump(docs, out)


if __name__ == "__main__":
    main()
