#!/usr/bin/python3.7
"""Preprocessing script

Script for preprocessing. Call with path to csv
"""

import pandas as pd
import numpy as np
import pickle
import click

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords")

class Preprocessing:
    
    def __init__(self):        
        pass
    
    def __call__(self, path_to_csv):
        self.path_to_csv = path_to_csv
        
        return self.preprocess(self.path_to_csv)
        
    def preprocess(self, path_to_csv):
        
        df = pd.read_csv(path_to_csv)
        
        russian_stopwords = stopwords.words("russian")
        docs = np.array(df['steps'])
        
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = tokenizer.tokenize(str(docs[idx]))  

        docs = [[token for token in doc if not token.isdigit()] for doc in docs]
        docs = [[token for token in doc if len(token) > 3] for doc in docs]
        docs = [[token for token in doc if token not in russian_stopwords] for doc in docs] 
        
        with open('docs.pkl', 'wb') as f: 
            pickle.dump(docs, f)
        
        
@click.command()
@click.option('-p', '--path_to_csv', required=True, type=str)
def main(path_to_csv):
    preprocesser = Preprocessing()
    preprocesser(path_to_csv)    

if __name__ == '__main__':
    main()
