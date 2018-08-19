#run the file below to make functions available
execfile('/Users/ai/Desktop/TextAnalytics/Class3/normalization.py')
import nltk
nltk.download('wordnet')
from gensim import corpora, models 
import numpy as np
import pandas as pd


#pretty printing function for topic exploration after training LDA
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
        
#toy corpus - example from class
toy_corpus = ["The fox jumps over the dog", 
              "The fox is very clever and quick", 
              "The dog is slow and lazy", 
              "The cat is smarter than the fox and the dog but it can never learn Java", 
              "Python is an excellent programming language", 
              "Java and Ruby are other programming languages", 
              "Python and Java are very popular programming languages", 
              "Python programs are smaller than Java programs"]

#define the function: it starts with normalizing and tokenizing your corpus and end with defining and LDA model on your data
def train_lda_model_gensim(corpus, total_topics=2): 
    norm_tokenized_corpus = normalize_corpus(corpus, tokenize=True) #normalize
    dictionary = corpora.Dictionary(norm_tokenized_corpus)          #create a dictionary for your corpus
    corpus_bow = [dictionary.doc2bow(text) for text in norm_tokenized_corpus] #create bag of words
    
    # tfidf = models.TfidfModel(corpus_bow)
    # corpus_tfidf = tfidf[corpus_bow]
    
    lda = models.LdaModel(corpus_bow, id2word = dictionary, iterations=1000, num_topics=total_topics) #define model
    return lda

#learn topics in the toy corpus
lda_gensim = train_lda_model_gensim(toy_corpus, total_topics=2)

#print the dsicovered topics (first 10 words in a topic with the highest weights)
print_topics_gensim(topic_model=lda_gensim, total_topics=2, num_terms=10, display_weights=True)





