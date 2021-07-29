# DVC 
___

Repository for the second task. Scripts perform topic modeling using Gensim LDA model on reciepts dataset. Evaluation performed using UMass metric. Also created visualization using pyldavis.

Make sure you are in the root of the cloned repo and use
```
dvc repro
```
to reproduce DVC pipeline.

**visualisation.html** **scores.json** creates at root folder

To see metrics use
```
dvc metrics show/diff
```
