#TfidfVectorizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from sklearn.metrics import confusion_matrix

def load_and_parse_data():
    chalga_lyrics = pd.read_json('a.json', encoding = 'utf-8')
    rap_lyrics = pd.read_json('b.json', encoding = 'utf-8')

    chalga_lyrics["type"] = 0
    rap_lyrics["type"] = 1

    print(chalga_lyrics.shape)
    print(rap_lyrics.shape)

    data = pd.concat([chalga_lyrics, rap_lyrics])
    data = data.sample(frac=1)
    #print(data.shape)

    print(data.head())
    #sns.countplot(x='type', data = data)
    plt.show()
    return data

def train_and_test_model(data):
    vectorizer = TfidfVectorizer()
    data_tfidf = vectorizer.fit_transform(data["lyrics"])

    K = [10, 100, 200, 300, 350]

    for k in K:
        data_tfidf_train = data_tfidf[:k]
        data_target_train = data[:k]
        data_tfidf_test = data_tfidf[k:]
        data_target_test = data[k:]

        # Logistic
        print('Logistic regression, # of train points:', k)
        clf = LogisticRegression(random_state=0).fit(data_tfidf_train, data_target_train["type"])
        print(clf.score(data_tfidf_test, data_target_test["type"]))

        #logistic confusion
        y_pred = clf.predict(data_tfidf_test)
        confusion = confusion_matrix(data_target_test['type'], y_pred)
        print(f"Confusion matrix is: \n", confusion)


            #SVM
        print('Support Vector Machines # of train points:', k)
        clf = svm.SVC()
        clf.fit(data_tfidf_train, data_target_train["type"])
        print(clf.score(data_tfidf_test, data_target_test["type"]))

            #SVM Confusion
        clf.fit(data_tfidf_train, data_target_train["type"])
        y_pred = clf.predict(data_tfidf_test)
        confusion = confusion_matrix(data_target_test['type'], y_pred)
        print(f"Confusion matrix is: \n", confusion)




data = load_and_parse_data()
train_and_test_model(data)
