import pandas as pd 
import numpy as np

# join list of token as single document string
import ast

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize


DATA = pd.read_csv("data/Text_Preprocessing.csv", usecols=["sentiment", "review_tokens_stemmed"])
DATA.columns = ["sentiment", "review"]

def joinTextList(texts):
    texts = ast.literal_eval(texts)
    return ' '.join([text for text in texts])

DATA["review_join"] = DATA["review"].apply(joinTextList)

print(DATA["review_join"].head())

# Menghitung TF-IDF menggunakan TfidfVectorizer
# banyaknya term yang akan digunakan, 
# di pilih berdasarkan top max_features 
# yang diurutkan berdasarkan term frequency seluruh corpus
max_features = 1000

# Feature Engineering 
print ("------- TF-IDF on Review data -------")

tf_idf = TfidfVectorizer(max_features=max_features, binary=True)
tfidf_mat = tf_idf.fit_transform(DATA["review_join"]).toarray()

print("\n TF-IDF ", type(tfidf_mat), tfidf_mat.shape)

terms = tf_idf.get_feature_names()

# sum tfidf frequency of each term through documents
sums = tfidf_mat.sum(axis=0)

# connecting term to its sums frequency
data = []
for col, term in enumerate(terms):
    data.append((term, sums[col] ))

ranking = pd.DataFrame(data, columns=['term','rank'])
ranking.sort_values('rank', ascending=False)

print(ranking)

max_features = 1000


def generateTfidfMat(min_gram, max_gram):
    cvect = CountVectorizer(max_features=max_features, ngram_range=(min_gram, max_gram))
    counts = cvect.fit_transform(DATA["review_join"])

    normalized_counts = normalize(counts, norm='l1', axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(min_gram, max_gram), smooth_idf=False)
    tfs = tfidf.fit_transform(DATA["review_join"])

    tfidf_mat = normalized_counts.multiply(tfidf.idf_).toarray()
    
    TF = normalized_counts.toarray()
    IDF = tfidf.idf_
    TF_IDF = tfidf_mat
    return TF, IDF, TF_IDF, tfidf.get_feature_names()

# ngram_range (1, 1) to use unigram only
tf_mat_unigram, idf_mat_unigram, tfidf_mat_unigram, terms_unigram = generateTfidfMat(1,1)

# ngram_range (2, 2) to use bigram only
tf_mat_bigram, idf_mat_bigram, tfidf_mat_bigram, terms_bigram = generateTfidfMat(2,2)

# ngram_range (3, 3) to use trigram only
tf_mat_trigram, idf_mat_trigram, tfidf_mat_trigram, terms_trigram = generateTfidfMat(3,3)

# ---------- check sparse data -------------------
idx_sample = 0

print("\nShow TFIDF sample ke-" + str(idx_sample), "\n")
print(DATA["review"][idx_sample], "\n")

print("\t\t\t", "TF", "\t\t", "IDF", "\t\t", "TF-IDF", "\t", "Term\n")
for i, item in enumerate(zip(tf_mat_unigram[idx_sample], idf_mat_unigram, tfidf_mat_unigram[idx_sample], terms_unigram)):
    if(item[2] != 0.0):
        print ("array position " + str(i) + "\t", 
               "%.6f" % item[0], "\t", 
               "%.6f" % item[1], "\t", 
               "%.6f" % item[2], "\t", 
               item[3])

def getTFUnigram(row):
    idx = row.name
    return [tf for tf in tf_mat_unigram[idx] if tf != 0.0]

DATA["TF_UNIGRAM"] = DATA.apply(getTFUnigram, axis=1)

def getIDFUnigram(row):
    idx = row.name
    return [item[1] for item in zip(tf_mat_unigram[idx], idf_mat_unigram) if item[0] != 0.0]

DATA["IDF_UNIGRAM"] = DATA.apply(getIDFUnigram, axis=1)

def getTFIDFUnigram(row):
    idx = row.name
    return [tfidf for tfidf in tfidf_mat_unigram[idx] if tfidf != 0.0]

DATA["TFIDF_UNIGRAM"] = DATA.apply(getTFIDFUnigram, axis=1)

DATA[["review", "TF_UNIGRAM", "IDF_UNIGRAM", "TFIDF_UNIGRAM"]].head()

# save TFIDF Unigram to Excel

DATA[["review", "TF_UNIGRAM", "IDF_UNIGRAM", "TFIDF_UNIGRAM"]].to_excel("data/TFIDF_Unigram.xlsx")

def getTFBigram(row):
    idx = row.name
    return [tf for tf in tf_mat_bigram[idx] if tf != 0.0]

DATA["TF_BIGRAM"] = DATA.apply(getTFBigram, axis=1)

def getIDFBigram(row):
    idx = row.name
    return [item[1] for item in zip(tf_mat_bigram[idx], idf_mat_bigram) if item[0] != 0.0]

DATA["IDF_BIGRAM"] = DATA.apply(getIDFBigram, axis=1)

def getTFIDFBigram(row):
    idx = row.name
    return [tfidf for tfidf in tfidf_mat_bigram[idx] if tfidf != 0.0]

DATA["TFIDF_BIGRAM"] = DATA.apply(getTFIDFBigram, axis=1)

def getTermBigram(row):
    idx = row.name
    return [item[1] for item in zip(tf_mat_bigram[idx], terms_bigram) if item[0] != 0.0]

DATA["review_BIGRAM"] = DATA.apply(getTermBigram, axis=1)

DATA[["review_BIGRAM", "TF_BIGRAM", "IDF_BIGRAM", "TFIDF_BIGRAM"]].head()


# save TFIDF Bigram to Excel

DATA[["review_BIGRAM", "TF_BIGRAM", "IDF_BIGRAM", "TFIDF_BIGRAM"]].to_excel("data/TFIDF_Bigram.xlsx")

def getTFTrigram(row):
    idx = row.name
    return [tf for tf in tf_mat_trigram[idx] if tf != 0.0]

DATA["TF_trigram"] = DATA.apply(getTFTrigram, axis=1)

def getIDFTrigram(row):
    idx = row.name
    return [item[1] for item in zip(tf_mat_trigram[idx], idf_mat_trigram) if item[0] != 0.0]

DATA["IDF_trigram"] = DATA.apply(getIDFTrigram, axis=1)

def get_TFIDF_trigram(row):
    idx = row.name
    return [tfidf for tfidf in tfidf_mat_trigram[idx] if tfidf != 0.0]

DATA["TFIDF_trigram"] = DATA.apply(get_TFIDF_trigram, axis=1)

def get_Term_trigram(row):
    idx = row.name
    return [item[1] for item in zip(tf_mat_trigram[idx], terms_trigram) if item[0] != 0.0]

DATA["review_TRIGRAM"] = DATA.apply(get_Term_trigram, axis=1)

DATA[["review_TRIGRAM", "TF_trigram", "IDF_trigram", "TFIDF_trigram"]].head()


# save TFIDF Trigram to Excel

DATA[["review_TRIGRAM", "TF_trigram", "IDF_trigram", "TFIDF_trigram"]].to_excel("data/TFIDF_Trigram.xlsx")