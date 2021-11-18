import pandas as pd 
import numpy as np

import ast

DATA = pd.read_csv("data/Text_Preprocessing.csv", usecols=["sentiment", "review_tokens_stemmed"])
DATA.columns = ["sentiment", "review"]

def convertTextList(texts):
    texts = ast.literal_eval(texts)
    return [text for text in texts]

DATA["review_list"] = DATA["review"].apply(convertTextList)

print(DATA["review_list"][10])

print("\ntype : ", type(DATA["review_list"][10]))


# hitung tf
def calcTF(document):
    # Counts the number of times the word appears in review
    TF_dict = {}
    for term in document:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1
    # Computes tf for each word
    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(document)
    return TF_dict

DATA["TF_dict"] = DATA['review_list'].apply(calcTF)

print(DATA["TF_dict"].head())

# Check TF result
index = 90

print('%20s' % "term", "\t", "TF\n")
for key in DATA["TF_dict"][index]:
    print('%20s' % key, "\t", DATA["TF_dict"][index][key])

# hitung DF
 
def calcDF(tfDict):
    count_DF = {}
    # Run through each document's tf dictionary and increment countDict's (term, doc) pair
    for document in tfDict:
        for term in document:
            if term in count_DF:
                count_DF[term] += 1
            else:
                count_DF[term] = 1
    return count_DF

DF = calcDF(DATA["TF_dict"])

#  hitung IDF

n_document = len(DATA)

def calcIDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict
  
#Stores the idf dictionary
IDF = calcIDF(n_document, DF)


#calc TF-IDF
def calcTFIDF(TF):
    TF_IDF_Dict = {}
    #For each word in the review, we multiply its tf and its idf.
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict

#Stores the TF-IDF Series
DATA["TF-IDF_dict"] =DATA["TF_dict"].apply(calcTFIDF)

# Check TF-IDF result
index = 90

print('%20s' % "term", "\t", '%10s' % "TF", "\t", '%20s' % "TF-IDF\n")
for key in DATA["TF-IDF_dict"][index]:
    print('%20s' % key, "\t", DATA["TF_dict"][index][key] ,"\t" , DATA["TF-IDF_dict"][index][key])

# sort descending by value for DF dictionary 
sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)[:50]

# Create a list of unique words from sorted dictionay `sorted_DF`
unique_term = [item[0] for item in sorted_DF]

def calcTFIDFVec(__TF_IDF_Dict):
    TF_IDF_vector = [0.0] * len(unique_term)

    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector

DATA["TF_IDF_Vec"] = DATA["TF-IDF_dict"].apply(calcTFIDFVec)

print("print first row matrix TF_IDF_Vec Series\n")
print(DATA["TF_IDF_Vec"][0])

print("\nmatrix size : ", len(DATA["TF_IDF_Vec"][0]))

# Convert Series to List
TF_IDF_Vec_List = np.array(DATA["TF_IDF_Vec"].to_list())

# Sum element vector in axis=0 
sums = TF_IDF_Vec_List.sum(axis=0)

data = []

for col, term in enumerate(unique_term):
    data.append((term, sums[col]))
    
ranking = pd.DataFrame(data, columns=['term', 'rank'])
ranking.sort_values('rank', ascending=False)

print(ranking)