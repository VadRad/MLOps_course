#!/usr/bin/env bash

python preprocessing.py -p 'recipes.csv' &&
python modeling.py -p 'docs.pkl' &&
python predict.py -m 'model' -p 'corpus.pkl' &&
