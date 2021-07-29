#!/usr/bin/python3
"""Modeling script

Script for Gensim LDA model. num_of_topics and model selection in topic-modeling notebook
Class call saves model and bigrams
"""

import pickle
import click
import io
import os
import yaml

from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases
from gensim.test.utils import datapath

params = yaml.safe_load(open("params.yaml"))["modeling"]


class Modeling:
    def __init__(self):        
        pass
    
    def __call__(self, path_to_docs):
        self.path_to_docs = path_to_docs
        
        return self.model(self.path_to_docs)    
    
    def model(self, path_to_docs):
        
        with open(path_to_docs, 'rb+') as f: 
            docs = pickle.load(f)      
        bigram = Phrases(docs, min_count=10)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)
                    
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=10, no_above=0.2)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        
        num_topics=params["num_topics"]
        
        model_lda=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

        return model_lda, corpus, dictionary
        
@click.command()
@click.option('-p', '--path_to_docs', required=True, type=str)
def main(path_to_docs):
    os.makedirs(os.path.join("data", "model"), exist_ok=True)
    modellda = Modeling()
    model, corpus, dictionary = modellda(path_to_docs)
    output_model = os.path.join("data", "model", "model.pkl")
    with io.open(output_model, "wb") as out: 
        pickle.dump(model, out)
    output_corpus = os.path.join("data", "model", "corpus.pkl")
    with io.open(output_corpus, "wb") as out: 
        pickle.dump(corpus, out)
    output_dictionary = os.path.join("data", "model", "dictionary.pkl")
    with io.open(output_dictionary, "wb") as out: 
        pickle.dump(dictionary, out)

if __name__ == '__main__':
    main()
