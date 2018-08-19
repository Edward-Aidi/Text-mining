# NEWS Classification

# Load the dataset of 18,000 news posts on 20 topics
# It's part of the sklearn dataset collections
# Posts are devided into "train" and "test"
# See more about the dataset at http://scikit-learn.org/stable/datasets/twenty_newsgroups.html 

import numpy as np
from sklearn import metrics

# load data
from sklearn.datasets import fetch_20newsgroups 

# select the training dataset only and posts on 4 select categories
# also, remove headers, footers, quotes
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
twenty_train = fetch_20newsgroups(categories = categories,
                                  subset='train', 
                                  shuffle=True, 
                                  remove=('headers', 'footers', 'quotes')) 
twenty_test = fetch_20newsgroups(categories = categories,
                                 subset='test', 
                                 shuffle=True,
                                 remove=('headers', 'footers', 'quotes')) 

#inspect the train data
twenty_train.data[0]         # 1st news post
twenty_train.target_names    # all unique categories
twenty_train.target[:10]     # first 10 categories
twenty_train.filenames.shape # number of news posts


# "bag of words" features
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range = (1,1))
X_train_counts = count_vect.fit_transform(twenty_train.data) # create document-term matrix 
X_train_counts.shape # dimensions: number of posts x number of features (words)
X_train_counts.nnz   # number of non-zero elements (nnz)
X_train_counts.nnz/float(X_train_counts.shape[0]) # average number of features present, per post
# compute the share of non-zero features (it is very small, very sparse matrix!)
100 * X_train_counts.nnz / (float(X_train_counts.shape[0]) * float(X_train_counts.shape[1]))

# PLEASE, COMPLETE: Your "bag of words" features for test data are here



# TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(ngram_range = (1,1))
X_train_tfidf = tfidf_vect.fit_transform(twenty_train.data)
X_train_tfidf.shape
X_train_tfidf.nnz/float(X_train_tfidf.shape[0]) # average number of features present, per post

# create TF-IDF features for test data
X_test_tfidf = tfidf_vect.transform(twenty_test.data)
X_test_tfidf.shape # dimensions: number of posts x number of features (words)
X_test_tfidf.nnz   # number of non-zero elements (nnz)
X_test_tfidf.nnz/float(X_test_tfidf.shape[0]) # average number of features present, per post
# share of non-zero features is very small, very sparse matrix!
100 * X_test_tfidf.nnz / (float(X_test_tfidf.shape[0]) * float(X_test_tfidf.shape[1]))



#NAIVE BAYES CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=.01)
clf.fit(X_train_tfidf, twenty_train.target)
predicted_nb = clf.predict(X_test_tfidf)

# evaluate the model
np.mean(predicted_nb == twenty_test.target) #accuracy rate 
metrics.confusion_matrix(twenty_test.target, predicted_nb) #full confusion matrix
metrics.accuracy_score(twenty_test.target, predicted_nb) #altenative way to compute accuracy

# PLEASE, COMPLETE: Your Naive Bayes with "bag of words" features is here



# SUPPORT VECTOR MACHINES CLASSIFIER (SVM)
from sklearn import linear_model
# linear_model.SGDClassifier with loss='hinge' defines SVM
clf_svm = linear_model.SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42) #define classifier
clf_svm.fit(X_train_tfidf, twenty_train.target) # train (fit) the model
predicted_svm = clf_svm.predict(X_test_tfidf)   # compute predicted values

#valuate SVM classifier
metrics.confusion_matrix(twenty_test.target, predicted_svm)
metrics.accuracy_score(twenty_test.target, predicted_svm) #altenative way to compute accuracy


# LOGIT_BASED CLASSIFIER
clf_log = linear_model.SGDClassifier(loss='log', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
clf_log.fit(X_train_tfidf, twenty_train.target)
predicted_log = clf_log.predict(X_test_tfidf)

# evaluate logit-based classifier
metrics.accuracy_score(twenty_test.target, predicted_log) #altenative way to compute accuracy
metrics.confusion_matrix(twenty_test.target, predicted_log)
