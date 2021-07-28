#!/usr/bin/python3.7
"""Modeling script

Script for Gensim LDA model. num_of_topics and model selection in topic-modeling notebook
Class call saves model and bigrams
"""

import pickle
import click

from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases
from gensim.test.utils import datapath

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
        
        num_topics=14
        
        model_lda=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        temp_file = datapath("model")
        model_lda.save(temp_file)
        
        with open('corpus.pkl', 'wb') as f: 
            pickle.dump(corpus, f)
        
        
@click.command()
@click.option('-p', '--path_to_docs', required=True, type=str)
def main(path_to_docs):
    modellda = Modeling()
    modellda(path_to_docs)    

if __name__ == '__main__':
    main()
