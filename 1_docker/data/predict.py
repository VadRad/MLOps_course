#!/usr/bin/python3.7
"""Predict script

Script for making predictions
Class call saves dataframe with id and topic number
"""

import pandas as pd
import pickle
import click

from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath

class Predicting:
    def __init__(self):        
        pass
    
    def __call__(self, path_to_model, path_to_corpus):
        self.path_to_model = path_to_model
        self.path_to_corpus = path_to_corpus
        
        return self.predict(self.path_to_model, self.path_to_corpus)    
    
    def predict(self, path_to_model, path_to_corpus):
        
        with open(path_to_corpus, 'rb') as f: 
            corpus = pickle.load(f)  
            
        temp_file = datapath(path_to_model)
        model_lda = LdaModel.load(temp_file)
        list_of_topics = []
        for doc in corpus:
            topics = model_lda.get_document_topics(doc)
            res = [[ i for i, j in topics ],
                   [ j for i, j in topics ]]    
            doc_topic = res[0][res[1].index(max(res[1]))]
            list_of_topics.append(doc_topic)
            
        df = pd.DataFrame(list_of_topics, columns=['topic'])
        df.to_csv('predictions.csv')
        
        
@click.command()
@click.option('-m', '--path_to_model', required=True, type=str)
@click.option('-p', '--path_to_corpus', required=True, type=str)
def main(path_to_model, path_to_corpus):
    predictor = Predicting()
    predictor(path_to_model, path_to_corpus)

if __name__ == '__main__':
    main()
