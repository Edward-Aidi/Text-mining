# SENTIMENT ANALYSIS AND MOIDEL EVALUATION: HELP FUNCTIONS

from HTMLParser import HTMLParser
import pandas as pd
import glob, os  
import numpy as np

class MLStripper(HTMLParser): 
    def __init__(self): 
        self.reset() 
        self.fed = [] 
    def handle_data(self, d): 
        self.fed.append(d) 
    def get_data(self): 
        return ' '.join(self.fed) 

def strip_html(text):
    html_stripper = MLStripper() 
    html_stripper.feed(text) 
    return html_stripper.get_data()

def normalize_accented_characters(text):
    text = unicodedata.normalize('NFKD', text.decode('utf-8') ).encode('ascii', 'ignore') 
    return text


def normalize_corpus(corpus, 
                     lemmatize=True, 
                     only_text_chars=False, 
                     tokenize=False):
    normalized_corpus = []     
    for index, text in enumerate(corpus): 
        text = normalize_accented_characters(text) 
        text = html_parser.unescape(text) 
        text = strip_html(text) 
        text = expand_contractions(text, CONTRACTIONS_MAP) 
        if lemmatize: 
            text = lemmatize_text(text) 
        else: 
            text = text.lower() 
            text = remove_special_characters(text) 
            text = remove_stopwords(text) 
        if only_text_chars: 
            text = keep_text_characters(text) 
        if tokenize: 
            text = tokenize_text(text) 
            normalized_corpus.append(text) 
        else: 
            normalized_corpus.append(text) 
    return normalized_corpus


        
        
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
def build_feature_matrix(documents, 
                         feature_type='frequency', 
                         ngram_range=(1, 1), 
                         min_df=0.0, 
                         max_df=1.0): 
    feature_type = feature_type.lower().strip()   
    if feature_type == 'binary': 
        vectorizer = CountVectorizer(binary=True, 
                                     min_df=min_df, 
                                     max_df=max_df, 
                                     ngram_range=ngram_range) 
    elif feature_type == 'frequency': 
        vectorizer = CountVectorizer(binary=False, 
                                     min_df=min_df, 
                                     max_df=max_df, 
                                     ngram_range=ngram_range) 
    elif feature_type == 'tfidf': 
        vectorizer = TfidfVectorizer(min_df=min_df, 
                                     max_df=max_df, 
                                     ngram_range=ngram_range) 
    else: 
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'") 
        
    feature_matrix = vectorizer.fit_transform(documents).astype(float) 
    return vectorizer, feature_matrix

from sklearn import metrics 

def display_evaluation_metrics(true_labels, predicted_labels, positive_class=1): 
    print 'Accuracy:', np.round( metrics.accuracy_score(true_labels, predicted_labels), 2) 
    print 'Precision:', np.round( metrics.precision_score(true_labels, predicted_labels, pos_label=positive_class, average='binary'), 2) 
    print 'Recall:', np.round( metrics.recall_score(true_labels, predicted_labels, pos_label=positive_class, average='binary'), 2) 
    print 'F1 Score:', np.round( metrics.f1_score(true_labels, predicted_labels, pos_label=positive_class, average='binary'), 2)


def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]): 
    cm = metrics.confusion_matrix(y_true=true_labels, 
                                  y_pred=predicted_labels, 
                                  labels=classes) 
    cm_frame = pd.DataFrame(data=cm, 
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 
                                                  labels=[[0,0],[0,1]]), 
                                                  index=pd.MultiIndex(levels=[['Actual:'], classes], 
                                                                      labels=[[0,0],[0,1]])) 
    print cm_frame

def display_classification_report(true_labels, predicted_labels, classes=[1,0]):
    report = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels, labels=classes)
    print report
