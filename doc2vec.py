#doc2Vec model Chalga ili rap

import gensim
import json
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import svm

def tagdocs(row):
    return TaggedDocument(row['lyrics'].split(" "), tags=[row['rownum']])

def docvecs(row):
    return model.infer_vector(row['lyrics'].split(" "))

data0 = pd.read_json('a.json', encoding = 'utf-8')
data0['type'] = 0
data1 = pd.read_json('b.json', encoding = 'utf-8')
data1['type'] = 1

data = pd.concat([data, data1])

data["rownum"] = data.index
data["taggeddoc"] = data.apply( tagdocs, axis=1)

data = data.sample(frac=1)
data_train = data[0:300]
data_test = data[300:]

# print(data)
# print(data_train.head(3))
# print(data_train["taggeddoc"].tolist())
# model = Doc2Vec(data_train["taggeddoc"].tolist(), vector_size=100, window=2, min_count=1, workers=4)
# model.save('chalga_ili_rap.model')

model = Doc2Vec.load('chalga_ili_rap.model')
#vector = model.infer_vector('любов ли е '.split(' '))
# print(vector)

data_train["docvecs"] = data_train.apply( docvecs, axis=1)
data_test["docvecs"] = data_test.apply( docvecs, axis=1)

print(data_train["docvecs"].head(3))
# Logistic regression
clf_logreg = LogisticRegression(random_state=0).fit(data_train["docvecs"].tolist(), data_train['type'])

# SVM
clf = svm.SVC()
clf.fit(data_train["docvecs"].tolist(), data_train['type'])

data_test['y_pred'] = clf.predict(data_test["docvecs"].tolist())
print('Support vectors')
print(classification_report(data_test['type'], data_test['y_pred']))

print('Logistic regression')
data_test['y_pred'] = clf_logreg.predict(data_test["docvecs"].tolist())
print(classification_report(data_test['type'], data_test['y_pred']))
# data_test.to_csv('result.csv')
