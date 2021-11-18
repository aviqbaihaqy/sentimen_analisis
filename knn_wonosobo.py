# Importing Essentials
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# random urutan dan split ke data training dan test
from sklearn.model_selection import train_test_split

DATA = pd.read_csv("data/Dataset_Review_Dieng.csv",
                   header=None, names=['sentiment', 'review'])

X_train, X_test, y_train, y_test = train_test_split(
    DATA['review'], DATA['sentiment'], test_size=0.2, random_state=123)

print("\nData training:")
print(len(X_train))
print(Counter(y_train))

print("\nData testing:")
print(len(X_test))
print(Counter(y_test))

# coba prediksi data baru
# review_baru = ['Dieng sekarang kotor']
review_baru = ['Dieng pemandangannya bagus sekali']

# transform ke tfidf dan train dengan KNN
KNN = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('knn', KNeighborsClassifier(n_neighbors = 3))])
KNN.fit(X_train, y_train)

#Accuracy using KNN Model
knn_pred = KNN.predict(review_baru)
print("Hasil prediksi {}".format(knn_pred))

# hitung akurasi data test
pred = KNN.predict(X_test)
akurasi = np.mean(pred == y_test)
print("Akurasi: {}".format(akurasi))

