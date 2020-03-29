#bag of words

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import WordCloud
from stop_words import get_stop_words

stop_words = get_stop_words('bg')

def load_and_parse_data():
    chalga_lyrics = pd.read_json('a.json', encoding = 'utf-8')
    rap_lyrics = pd.read_json('b.json', encoding = 'utf-8')

    chalga_lyrics["type"] = 0
    rap_lyrics["type"] = 1


    data = pd.concat([chalga_lyrics, rap_lyrics])
    data = data.sample(frac=1)

    #print(data.head())
    return data

def train_and_test_model(data):
    vectorizer = CountVectorizer()
    data_bow = vectorizer.fit_transform(data["lyrics"])

    #print(vectorizer.get_feature_names())

    KK = [10, 100, 200, 300, 350] #size of train model
    print('Shape of dataset is:', data_bow.shape)

    for K in KK:
        data_bow_train = data_bow[:K]
        data_target_train = data[:K]
        data_bow_test = data_bow[K:]
        data_target_test = data[K:]


        # Logistic
        print('Logistic regression of train points:', K)
        clf = LogisticRegression(random_state=0).fit(data_bow_train, data_target_train["type"])
        print(clf.score(data_bow_test, data_target_test["type"]))

        #SVM
        print('Support Vector Machines # of train points:', K)
        clf = svm.SVC()
        clf.fit(data_bow_train, data_target_train["type"])
        print(clf.score(data_bow_test, data_target_test["type"]))

data = load_and_parse_data()
train_and_test_model(data)


# Start with one review:
text = data_bow.description[1]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
print(plt.show())
