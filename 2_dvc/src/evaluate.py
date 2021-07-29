#!/usr/bin/python3
"""Modeling script

Script for Gensim LDA model. num_of_topics and model selection in topic-modeling notebook
Class call saves model and bigrams
"""

import pickle
import click
import json
import io
import os
import yaml

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim_models

class Modeling:
    def __init__(self):        
        pass
    
    def __call__(self, path_to_model, path_to_docs, path_to_corpus, path_to_dictionary):
        self.path_to_model = path_to_model
        self.path_to_corpus = path_to_corpus
        self.path_to_docs = path_to_docs
        self.path_to_dictionary = path_to_dictionary
        
        return self.evaluate(self.path_to_model, self.path_to_docs, self.path_to_corpus, self.path_to_dictionary)    
    
    def evaluate(self, path_to_model, path_to_docs, path_to_corpus, path_to_dictionary):
        
        with open(path_to_model, 'rb+') as f: 
            model = pickle.load(f)
        with open(path_to_corpus, 'rb+') as f: 
            corpus = pickle.load(f)
        with open(path_to_docs, 'rb+') as f: 
            docs = pickle.load(f)   
        with open(path_to_dictionary, 'rb+') as f: 
            dictionary = pickle.load(f)         
        coherence_model_lda = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence="u_mass")
        coherence_lda = coherence_model_lda.get_coherence()

        vis_data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)

        return coherence_lda, vis_data
        
@click.command()
@click.option('-m', '--path_to_model', required=True, type=str)
@click.option('-p', '--path_to_docs', required=True, type=str)
@click.option('-c', '--path_to_corpus', required=True, type=str)
@click.option('-d', '--path_to_dictionary', required=True, type=str)
def main(path_to_model, path_to_docs, path_to_corpus, path_to_dictionary):
    modellda = Modeling()
    score, vis = modellda(path_to_model, path_to_docs, path_to_corpus, path_to_dictionary)
    scores_file = os.path.join("scores.json")
    with open(scores_file, "w") as fd:
        json.dump({"UMass": score}, fd)
    vis_path = os.path.join("visualization.html")
    pyLDAvis.save_html(vis, vis_path)

if __name__ == '__main__':
    main()
