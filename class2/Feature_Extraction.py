# pip install pattern
# pip install metrics

import numpy as np
import pandas as pd

# Create a toy training corpus and a toy testing corpus
CORPUS_train = ['the sky is blue', 'sky is blue and sky is beautiful', 'the beautiful sky is so blue', 'i love blue cheese' ] 
CORPUS_test = ['loving this blue sky today']

# define function that creates a "bag of words" (bow)
# function parameters:
# - min_df=1: words with minimum count = 1 are incleded as features
# - ngram_range = (1,1): include unigrams only [(1,2) would include unigrams and bigrams]
from sklearn.feature_extraction.text import CountVectorizer
def bow_extractor(corpus, ngram_range=(1,1)): 
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range) 
    features = vectorizer.fit_transform(corpus) 
    return vectorizer, features

# define function to make a pretty table for features and weights
def display_features(features, feature_names): 
    df = pd.DataFrame(data=features, columns=feature_names)
    print (df)
    
# apply the bow function and print the results
bow_vectorizer, bow_features = bow_extractor(CORPUS_train)
features = bow_features.todense()
feature_names = bow_vectorizer.get_feature_names()
display_features(features, feature_names)


# Define function that extracts features and assigns TF-IDF weights to them
from sklearn.feature_extraction.text import TfidfVectorizer 
def tfidf_extractor(corpus, ngram_range=(1,1)): 
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True,
                                 use_idf=True, ngram_range=ngram_range) 
    features = vectorizer.fit_transform(corpus) 
    return vectorizer, features


# build TF-IDF vectorizer and get feature vectors from the training corpus
tfidf_vectorizer, tdidf_features = tfidf_extractor(CORPUS_train)
display_features(np.round(tdidf_features.todense(), 2), feature_names)

# get TF-IDF feature vector for the "test" document
nd_tfidf = tfidf_vectorizer.transform(CORPUS_test)
display_features(np.round(nd_tfidf.todense(), 2), feature_names)



