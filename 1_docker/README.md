# Docker 
___

Repository for the first task. Scripts perform topic modeling using Gensim LDA model on reciepts dataset. 

### Folder Organization
------------

    ├── README.md                   <- The top-level README 
    │
    ├── Dockerfile                  <- Dockerfile to create docker images
    │ 
    ├── docker-compose.yml          <- Docker compose configuration files
    │
    └── data 
          └── requirements.txt        <- generated with `pip freeze > requirements.txt
          └── entrypoint.sh           <- entrypoint script running .py  files in container
          └── topicmodeling.ipynb     <- notebook with EDA and model selection
          └── recipes.csv             <- .csv file with dataset
          └── preprocessing.py        <- preprocessing script 
          └── modeling.py             <- train gensim LDA model
          └── predict.py              <- make predictions
Make sure you are in the root of the cloned repo and use
```
docker-compose-up
```
to build docker image, launch a container and create predictions. 

**predictions.csv** creates at data folder
