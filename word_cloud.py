import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from wordcloud import WordCloud
from stop_words import get_stop_words
import nltk

#from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()

stop_words = get_stop_words('bg') + ["'", "x4", "x2", "x3", "4x", "2x", "3x"]

chalga_lyrics = pd.read_json('a.json', encoding = 'utf-8')
rap_lyrics = pd.read_json('b.json', encoding = 'utf-8')

vectorizer = CountVectorizer()
data_bow = vectorizer.fit_transform(rap_lyrics["lyrics"]) #getting data into BOW form
words = vectorizer.get_feature_names()

#stemming words:
stemmed_words = ""
for word in words:
    stemmed += f" {lancaster.stem(word)}"

#print(stemmed)
#print(words)

# Creation and generation of a word cloud image (for rap):

cloud = WordCloud(stopwords='stop_words', background_color="white").generate(str(stemmed))

cloud = WordCloud(stopwords='stop_words', background_color="white").generate(str(words))

    # Display the generated image:
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
print(plt.show())
