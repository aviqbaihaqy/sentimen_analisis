# Importing Essentials
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA = pd.read_csv("data/Text_Preprocessing.csv", usecols=["review", "TFIDF_UNIGRAM"])
DATA.columns = ["review", "TFIDF_UNIGRAM"]


knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                     metric='cosine', metric_params=None, n_jobs=1)

knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print('KNN with TFIDF accuracy = ' + str(acc * 100) + '%')

scores = cross_val_score(knn, X_train, y_train, cv=3)
print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean(), scores.std() * 2))
print(scores)
