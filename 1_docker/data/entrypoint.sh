#!/usr/bin/env bash

$(command -v python) preprocessing.py -p recipes.csv
$(command -v python) modeling.py -p docs.pkl
$(command -v python) predict.py -m model -p corpus.pkl
