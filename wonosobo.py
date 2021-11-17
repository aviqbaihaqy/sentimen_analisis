import pandas as pd
import numpy as np
import re
import string

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from nltk.corpus import stopwords

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

path = 'data/dataset_film.csv'
data = pd.read_csv(path, header=None, names=['sentiment', 'review'])

# TEXT PREPROCESSING
# 1 case folding
# gunakan fungsi Series.str.lower() pada Pandas
data['review'] = data['review'].str.lower()

print('case folding result: \n')
print(data['review'].head())
print('\n\n\n')

# 2 tokenizing


def removereviewSpecial(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t', " ").replace(
        '\\n', " ").replace('\\u', " ").replace('\\', "")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
# end


data['review'] = data['review'].apply(removereviewSpecial)

# remove number


def removeNumber(text):
    return re.sub(r"\d+", "", text)


data['review'] = data['review'].apply(removeNumber)

# remove punctuation


def removePunctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


data['review'] = data['review'].apply(removePunctuation)

# remove whitespace leading & trailing


def removeWhitespaceLt(text):
    return text.strip()


data['review'] = data['review'].apply(removeWhitespaceLt)

# remove multiple whitespace into single whitespace


def removeWhitespaceMultiple(text):
    return re.sub('\s+', ' ', text)


data['review'] = data['review'].apply(removeWhitespaceMultiple)

# remove single char


def removeSingleChar(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


data['review'] = data['review'].apply(removeSingleChar)

# NLTK word rokenize


def word_tokenize_wrapper(text):
    return word_tokenize(text)


data['review_tokens'] = data['review'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n')
print(data['review_tokens'].head())
print('\n\n\n')

# NLTK calc frequency distribution


def freqDistWrapper(text):
    return FreqDist(text)


data['review_tokens_fdist'] = data['review_tokens'].apply(freqDistWrapper)

print('Frequency Tokens : \n')
print(data['review_tokens_fdist'].head().apply(lambda x: x.most_common()))

# # visualisai freq_token
# def visualiseFreqToken(data):
#     df_freq_tokens= pd.DataFrame.from_dict(data, orient='index')
#     df_freq_tokens.columns = ['Frequency']
#     df_freq_tokens.index.name = 'Key'
#     df_freq_tokens.plot(kind='bar')


# df_freq_tokens = data['review_tokens_fdist'].head().apply(visualiseFreqToken)

# 3 Filtering (Stopword Removal)
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')
# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv(
    "data/stopwords.txt", names=["stopwords"], header=None)

# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# convert list to dictionary
list_stopwords = set(list_stopwords)

# remove stopword pada list token


def stopwordsRemoval(words):
    return [word for word in words if word not in list_stopwords]


data['review_tokens_WSW'] = data['review_tokens'].apply(stopwordsRemoval)

print(data['review_tokens_WSW'].head())

# normalisasi
normalizad_word = pd.read_excel("data/normalisasi.xlsx", engine='openpyxl')

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]


def normalizedTerm(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


data['review_normalized'] = data['review_tokens_WSW'].apply(normalizedTerm)

print(data['review_normalized'].head())

# 4 stemming
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed


def stemmedWrapper(term):
    return stemmer.stem(term)


term_dict = {}

for document in data['review_normalized']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '

print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmedWrapper(term)
    print(term, ":", term_dict[term])

print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def getStemmedTerm(document):
    return [term_dict[term] for term in document]


data['review_tokens_stemmed'] = data['review_normalized'].swifter.apply(
    getStemmedTerm)
print(data['review_tokens_stemmed'])

data.to_csv("Text_Preprocessing.csv")
