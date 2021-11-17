import pandas as pd
import numpy as np
import re
import string

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

path = 'data/dataset_film.csv'
data = pd.read_csv(path,header=None,names=['sentiment','review'])

# TEXT PREPROCESSING
# 1 case folding
# gunakan fungsi Series.str.lower() pada Pandas
data['review'] = data['review'].str.lower()

print('case folding result: \n')
print(data['review'].head(5))
print('\n\n\n')

# 2 tokenizing
def removereviewSpecial(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
#end

data['review'] = data['review'].apply(removereviewSpecial)

#remove number
def removeNumber(text):
    return  re.sub(r"\d+", "", text)

data['review'] = data['review'].apply(removeNumber)

#remove punctuation
def removePunctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

data['review'] = data['review'].apply(removePunctuation)

#remove whitespace leading & trailing
def removeWhitespaceLt(text):
    return text.strip()

data['review'] = data['review'].apply(removeWhitespaceLt)

#remove multiple whitespace into single whitespace
def removeWhitespaceMultiple(text):
    return re.sub('\s+',' ',text)

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
print(data['review_tokens'].head(5))
print('\n\n\n')

# 3 filtering
# 4 stemming