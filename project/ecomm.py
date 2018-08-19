#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Di Ai, Ziwei Lu, Bingqing He, Haiyi Yu
"""

# E-commerce LDA model building
execfile('/Users/ai/Desktop/TextAnalytics/Class3/normalization.py')
execfile('/Users/ai/Desktop/TextAnalytics/class6/Model_Eval_Help.py')

import nltk
nltk.download('wordnet')
from gensim import corpora, models 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pyLDAvis.gensim
from itertools import chain
from sklearn.model_selection import train_test_split
from nltk.util import ngrams
from collections import Counter
from gensim.models import CoherenceModel 



# first read in the data and then perform different methods of feature engineering
ecomm_corpus = pd.read_csv('/Users/ai/Desktop/TextAnalytics/project/Womens Clothing E-Commerce Reviews.csv', delimiter=',')

ecomm_corpus.columns = ['id',
 'Clothing ID',
 'Age',
 'Title',
 'Review Text',
 'Rating',
 'Recommended IND',
 'Positive Feedback Count',
 'Division Name',
 'Department Name',
 'Class Name']

# select both id review text
ecomm_corpus_reviewid = ecomm_corpus[['id','Review Text']]
ecomm_corpus_review = ecomm_corpus.loc[:, ecomm_corpus.columns.str.startswith('Review')]

ecomm_corpus_review = ecomm_corpus[['Review Text']]
# drop colomns that has NA, original 23,486 ids and become 22,641 ids
ecomm_corpus_reviewid_no = ecomm_corpus_reviewid.dropna()
review_sample_id = random.sample(ecomm_corpus_reviewid_no['id'], 2000)

ecomm_corpus_review_lst = ecomm_corpus_review.ix[:, 0].tolist()
cleaned_ecomm_review = [x for x in ecomm_corpus_review_lst if str(x) != 'nan'] # delete nan in the original review corpus

# use a smaller sample 2k of the 22,641 data from the original data
random.seed(2018)
review_sample = random.sample(cleaned_ecomm_review, 2000)

# print lda function
# pretty printing function for topic exploration after training LDA
def print_topics_gensim(topic_model, total_topics=1,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):
    
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt,2)) 
                 for word, wt in topic 
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print
        
# lemmatize the corpus first
lemm_corpus = [] 
for index, text in enumerate(review_sample):
    text = lemmatize_text(text)
    lemm_corpus.append(text)

lemm_review = [x.encode('UTF8') for x in lemm_corpus]

# we could use three ways of feature selection to do extract the features of our sample data, one is bow- frequency
# and another one is tf-idf

# bow
def train_lda_model_gensim(corpus, total_topics=5): 
    norm_tokenized_corpus = normalize_corpus(corpus, lemmatize= False, tokenize=True) #normalize
    dictionary = corpora.Dictionary(norm_tokenized_corpus)          #create a dictionary for your corpus
    corpus_bow = [dictionary.doc2bow(text) for text in norm_tokenized_corpus] #create bag of words
    lda = models.LdaModel(corpus_bow, id2word = dictionary, iterations=1000, num_topics=total_topics) #define model
    return lda

# tf-idf
def train_lda_model_gensim_tfidf(corpus, total_topics=5): 
    norm_tokenized_corpus = normalize_corpus(corpus, lemmatize= False, tokenize=True) #normalize
    dictionary = corpora.Dictionary(norm_tokenized_corpus)          #create a dictionary for your corpus
    corpus_bow = [dictionary.doc2bow(text) for text in norm_tokenized_corpus] #create bag of words
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    lda = models.LdaModel(corpus_tfidf, id2word = dictionary, iterations=1000, num_topics=total_topics, alpha = 'auto') #define model
    return lda

# bi-gram
def train_lda_model_gensim_bi(corpus, total_topics=5): 
    norm_tokenized_corpus = normalize_corpus(review_sample, lemmatize= False, tokenize=True) #normalize
    bigrams = ngrams(norm_tokenized_corpus,2)
    dictionary = corpora.Dictionary(bigrams)          #create a dictionary for your corpus
    dictionary = corpora.Dictionary(norm_tokenized_corpus)  
    corpus_bow = [dictionary.doc2bow(text) for text in bigrams] #create bag of words
    lda = models.LdaModel(corpus_bow, id2word = dictionary, iterations=1000, num_topics=total_topics) #define model
    return lda

np.random.seed(2018)

#learn topics in the ecomm corpus
random.seed(2018)
lda_gensim_bow = train_lda_model_gensim(lemm_review, total_topics=5)
lda_gensim_tfidf = train_lda_model_gensim_tfidf(lemm_review, total_topics=5)
lda_gensim_bi = train_lda_model_gensim_bi(lemm_review, total_topics=5)

#print the dsicovered topics (first 5 words in a topic with the highest weights)
print_topics_gensim(topic_model=lda_gensim_bow, total_topics=5, num_terms=10, display_weights=True)
print_topics_gensim(topic_model=lda_gensim_tfidf, total_topics=5, num_terms=10, display_weights=True)
print_topics_gensim(topic_model=lda_gensim_bi, total_topics=5, num_terms=10, display_weights=True)

###############
# We will use tf-idf as the better feature extraction method
# we will assign each document a topic according to LDA result
topics_terms = lda.state.get_lambda()
print(lda_gensim_tfidf[corpus_tfidf[340]])

id2word = dictionary
mm = [id2word.doc2bow(text) for text in norm_tokenized_corpus]
lda_corpus = lda[mm]

# An average score will be used as threshold to determine which topic this sentence is in
scores = []
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print threshold

# assign topics for ids
    lda_corpus = [max(prob,key=lambda y:y[1])
                    for prob in lda[mm] ]
    playlists = [[] for i in xrange(5)]
    for i, x in enumerate(lda_corpus):
        playlists[x[0]].append(review_sample[i])

# Write out the csv
import csv
myFile = open('ldatopics.csv', 'w')
with myFile:  
   writer = csv.writer(myFile)
   writer.writerows(lda_corpus)
   
   
with open('review_sample.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in review_sample:
        writer.writerow([val]) 

##############
# define function to compute the coherenace score ...
# .... for a series of LDA models with different number of topics

ldatopics = lda.show_topics(formatted=False)

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(2, limit):
        lm = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(2, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.title('Topic Coherence Score as a function of the number of topics')
    plt.show()
    
    return lm_list, c_v

# compute and coherence score for LDA models with up to 20 topics (parameter limit = 20)
lmlist, c_v = evaluate_graph(dictionary=dictionary, 
                             corpus=corpus_bow, 
                             texts=norm_tokenized_corpus, 
                             limit=10)


"""
visualization of tf-idf feature selection LDA
"""

# compute and plot distribution of topics across the total documents in the corpus
ND = 2000
topic_dist = [sum(lda_gensim_tfidf[corpus_tfidf[doc]][topic][1] for doc in range(ND))/ND for topic in range(total_topics)]
plt.bar(range(total_topics),topic_dist)
plt.xticks(range(total_topics),range(total_topics));
plt.xlabel('Topic ids');
plt.title('Distribution of topics across the sampled '+ str(ND) + ' documents')
plt.show()


total_topics = 5

def print_topics(topic_model, total_topics = total_topics,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print 'Topic #'+str(index+1)+' with weights'
            print topic[:num_terms] if num_terms else topic
        else:
            print 'Topic #'+str(index+1)+' without weights'
            tw = [term for term, wt in topic]
            print tw[:num_terms] if num_terms else tw
        print


# visualize the topics and output it as a html page: 
# LOCATE lda.html file on your machine and the the interactive results!
vis = pyLDAvis.gensim.prepare(lda_gensim_tfidf, corpus_tfidf, dictionary)
pyLDAvis.save_html(vis, 'ecommerce_lda.html')

#########################################################################
# build classification model - SVM and Logit
train = pd.read_csv("/Users/ai/Desktop/train.csv")
train_reviews = np.array(train['Review.Text']) 
train_positive = np.array(train['Positive_Bi'])

test = pd.read_csv("/Users/ai/Desktop/test.csv")
test_reviews = np.array(test['Review.Text']) 
test_positive = np.array(test['Positive_Bi'])

# prepare training data for the recommended
norm_train_reviews = normalize_corpus(train_reviews, 
                                      lemmatize=True, 
                                      only_text_chars=True) 
norm_test_reviews = normalize_corpus(test_reviews, 
                                      lemmatize=True, 
                                      only_text_chars=True) 
# feature extraction: tf-idf                                                                          
test_features = vectorizer.transform(norm_test_reviews)

# feature extraction: tf-idf 
random.seed(2018)                                                                        
vectorizer, train_features = build_feature_matrix(documents=norm_train_reviews, 
                                                  feature_type='tfidf', 
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, 
                                                  max_df=1.0)

# SVM
from sklearn.linear_model import SGDClassifier 
svm = SGDClassifier(loss='hinge', max_iter=200)
svm.fit(train_features, train_positive)
svm_predicted_positive = svm.predict(test_features)    

display_evaluation_metrics(true_labels=test_positive,
                           predicted_labels=svm_predicted_positive,
                           positive_class=1)

display_confusion_matrix(true_labels=test_positive,
                         predicted_labels=svm_predicted_positive,
                         classes=[0, 1])

# Logit
logit = SGDClassifier(loss='log', max_iter=200)
logit.fit(train_features, train_positive)

# predict the usefulness for test dataset movie reviews (LOGIT)
logit_predicted_positive = logit.predict(test_features)        
# evaluate model prediction performance 

# show performance metrics (LOGIT)
display_evaluation_metrics(true_labels=test_positive,
                           predicted_labels=logit_predicted_positive,
                           positive_class= 1)
display_confusion_matrix(true_labels=test_positive,
                         predicted_labels=logit_predicted_positive,
                         classes=[0, 1])



#################################