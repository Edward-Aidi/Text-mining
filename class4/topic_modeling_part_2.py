#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:53:13 2018

@author: Di Ai
"""

import pandas as pd 
import numpy as np # load reviews 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel 
import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import pyLDAvis.gensim # DO!: pip install pyLDAvis


# DO!: change the path to for your machine
runfile('/Users/ai/Desktop/TextAnalytics/Class3/normalization.py')


# read corpus of ABC news headlines (focus on Australia)
# data source: https://www.kaggle.com/therohk/million-headlines/data
# DO!: change the path to for your machine
data = pd.read_csv('/Users/ai/Desktop/TextAnalytics/class4/abcnews-date-text.csv', 
                   error_bad_lines=False);
                   
data.shape                 #dimensions of the data
list(data.columns.values)  #column names

# We only need the Headlines text column from a subset of the data
NumDoc = 50000           
CORPUS = np.array(data['headline_text'])[0:NumDoc]
CORPUS.shape
print CORPUS[310]

# Set number of topics to discover
total_topics = 6

#pretty printing function for topic exploration after training LDA
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
        

# prep the text
norm_tokenized_corpus = normalize_corpus(CORPUS, tokenize=True) # normalize and tokenize
dictionary = corpora.Dictionary(norm_tokenized_corpus)          # create a dictionary 
corpus_bow = [dictionary.doc2bow(text) for text in norm_tokenized_corpus] # create the "bag-of-words" 


# run the LDA model  
lda = models.LdaModel(corpus_bow,                # corpus as "bag of words"
                      id2word = dictionary,      # dictionary
                      iterations=1000,           # max number of iterations
                      num_topics=total_topics,   # number of topics to identify
                      minimum_probability=0.001) # min probability to relate the word to the topic


# print the LDA results: identified topics
print_topics(topic_model=lda, 
             total_topics = total_topics, 
             num_terms=5, 
             display_weights=True)

# see topic distribution for a document # = 340
print(lda[corpus_bow[340]])
# topic shares sum to 1.
sum(lda[corpus_bow[340]][topic][1] for topic in range(total_topics))


# compute and plot distribution of topics across the first ND documents in the corpus
ND = 500
topic_dist = [sum(lda[corpus_bow[doc]][topic][1] for doc in range(ND))/ND for topic in range(total_topics)]
plt.bar(range(total_topics),topic_dist)
plt.xticks(range(total_topics),range(total_topics));
plt.xlabel('Topic ids');
plt.title('Distribution of topics across the first '+ str(ND) + ' documents')
plt.show()

#################################


# visualize the topics and output it as a html page: 
# LOCATE lda.html file on your machine and the the interactive results!
vis = pyLDAvis.gensim.prepare(lda,corpus_bow,dictionary)
pyLDAvis.save_html(vis, 'Ai, Di_lda.html')

# Relevancy weight parameter - λ (0 ≤ λ ≤ 1): 
# small λ highlights potentially rare, but exclusive terms for the selected topic;
# large values of λ (near 1) highlight frequent, but not necessarily exclusive, terms for the selected topic;
# Relevancy = λ log[p(term | topic)] + (1 - λ) log[p(term | topic)/p(term)]


# Additional information on how to use this visualization:
# http://www.kennyshirley.com/LDAvis/
# https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

#################################
# SELECTING NUMBER of TOPICS BASED on Coherence of Words in a Topic
# thank you to: Devashish Deshpande  
 
# Example for computing coherence for a toy
texts_example = [['human', 'interface', 'computer'],
                 ['survey', 'user', 'computer', 'system', 'response', 'time'],
                 ['eps', 'user', 'interface', 'system'],
                 ['system', 'human', 'system', 'eps'],
                 ['user', 'response', 'time'],
                 ['trees'],
                 ['graph', 'trees'],
                 ['graph', 'minors', 'trees'],
                 ['graph', 'minors', 'survey']]
dictionary_example = corpora.Dictionary(texts_example)
corpus_example = [dictionary.doc2bow(text) for text in texts_example]

# assume that below are the topics identified by the LDA model
topics_example = [['human', 'computer', 'system', 'interface'],
                  ['graph', 'minors', 'trees', 'eps']]

cm = CoherenceModel(topics=topics_example,
                    dictionary = dictionary_example,
                    texts=texts_example, 
                    coherence='c_v')
cm.get_coherence()


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
                             limit=20)





