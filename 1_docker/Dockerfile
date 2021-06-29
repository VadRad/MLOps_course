FROM python:3.7-slim-buster

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

WORKDIR /home/data/

RUN pip install jupyter -U && pip install jupyterlab

COPY ./data/requirements.txt requirements.txt
COPY ./data/preprocessing.py preprocessing.py
COPY ./data/modeling.py modeling.py
COPY ./data/predict.py predict.py
COPY ./data/recipes.csv recipes.csv

RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["sh", "entrypoint.sh"]

