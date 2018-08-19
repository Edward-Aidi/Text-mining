# SENTIMENT ANALYSIS, MODEL EVALUATION, BOOTSTRAPPING

# DO: CHANGE THE PATH!
execfile('/Users/ai/Desktop/TextAnalytics/Class3/normalization.py')
execfile('/Users/ai/Desktop/TextAnalytics/class6/Model_Eval_Help.py')

import pandas as pd
import glob, os  
import numpy as np

#PREPPING TRAIN DATA
# read the movie review data - train data
# Do: CHANGE PATH!
reviews_neg = glob.glob('/Users/ai/Desktop/TextAnalytics/class6/reviews/train/neg/*.txt')
d_neg = list()
for review_file in reviews_neg:
    with open(review_file) as f:
        review = os.path.basename(review_file)
        d_neg.append(pd.DataFrame({'file_name': review, 'review_text': f.readlines(), 'polarity': 'negative'}))

# Do: CHANGE PATH!
reviews_pos = glob.glob('/Users/ai/Desktop/TextAnalytics/class6/reviews/train/pos/*.txt')
d_pos = list()
for review_file in reviews_pos:
    with open(review_file) as f:
        review = os.path.basename(review_file)
        d_pos.append(pd.DataFrame({'file_name': review, 'review_text': f.readlines(), 'polarity': 'positive'}))
        
doc_neg = pd.concat(d_neg)
doc_pos = pd.concat(d_pos)

# combine training positive and negative reviews into one object
doc_train = pd.concat([doc_neg, doc_pos])
doc_train.shape
doc_train.head()

train_reviews = np.array(doc_train['review_text']) 
train_sentiments = np.array(doc_train['polarity'])
train_sentiments.shape

# normalization for training data 
norm_train_reviews = normalize_corpus(train_reviews, 
                                      lemmatize=True, 
                                      only_text_chars=True) 

# feature extraction for training data 
# feature type can be one the the three: frequency, tfidf, binary                                                                           
vectorizer, train_features = build_feature_matrix(documents=norm_train_reviews, 
                                                  feature_type='tfidf', 
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, 
                                                  max_df=1.0)


# PREPPING TEST DATA
# Do: CHANGE PATH!
test_reviews_neg = glob.glob('/Users/ai/Desktop/TextAnalytics/class6/reviews/test/neg/*.txt')
test_d_neg = list()
for review_file in test_reviews_neg:
    with open(review_file) as f:
        review = os.path.basename(review_file)
        test_d_neg.append(pd.DataFrame({'file_name': review, 'review_text': f.readlines(), 'polarity': 'negative'}))

# Do: CHANGE PATH!
test_reviews_pos = glob.glob('/Users/ai/Desktop/TextAnalytics/class6/reviews/test/pos/*.txt')
test_d_pos = list()
for review_file in test_reviews_pos:
    with open(review_file) as f:
        review = os.path.basename(review_file)
        test_d_pos.append(pd.DataFrame({'file_name': review, 'review_text': f.readlines(), 'polarity': 'positive'}))
        
test_doc_neg = pd.concat(test_d_neg)
test_doc_pos = pd.concat(test_d_pos)

#compbing test pos and test neg reviews into one object
doc_test = pd.concat([test_doc_neg, test_doc_pos])
doc_test.shape
doc_test.head()

test_reviews = np.array(doc_test['review_text']) 
test_sentiments = np.array(doc_test['polarity'])
test_sentiments.shape

# normalization for test reviews
norm_test_reviews = normalize_corpus(test_reviews, 
                                      lemmatize=True, 
                                      only_text_chars=True) 
# feature extraction for test reviews                                                                            
test_features = vectorizer.transform(norm_test_reviews)


# Selecting 10 reviews as sample reviews to have a loot at
sample_ids = [150, 250, 351, 450, 550, 661, 751, 851, 951, 995] 
sample_reviews = [(test_reviews[index], test_sentiments[index]) for index in sample_ids]


# TRAINING THE MODEL USNG SUPPORT VECTOR MACHINE ALGORITHM
from sklearn.linear_model import SGDClassifier 
svm = SGDClassifier(loss='hinge', max_iter=200)
svm.fit(train_features, train_sentiments)

# PREDICTIONS

# predictions for the sample reviews (SVM)
for doc_index in sample_ids:
    print 'Review:-'
    print test_reviews[doc_index]
    print 'Actual Labeled Sentiment:', test_sentiments[doc_index]
    doc_features = test_features[doc_index]
    predicted_sentiment = svm.predict(doc_features)[0]
    print 'Predicted Sentiment:', predicted_sentiment    
    print

# predict the sentiment for test dataset movie reviews  (SVM)
svm_predicted_sentiments = svm.predict(test_features)        
# evaluate model prediction performance 

# show performance metrics (SVM)
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=svm_predicted_sentiments,
                           positive_class='positive')

# show confusion matrix (SVM)
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=svm_predicted_sentiments,
                         classes=['positive', 'negative'])

# show detailed per-class classification report (SVM)
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=svm_predicted_sentiments,
                              classes=['positive', 'negative'])


# TRAIN A Logit classifier
logit = SGDClassifier(loss='log', max_iter=200)
logit.fit(train_features, train_sentiments)

# predict the sentiment for test dataset movie reviews (LOGIT)
logit_predicted_sentiments = logit.predict(test_features)        
# evaluate model prediction performance 

# show performance metrics (LOGIT)
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=logit_predicted_sentiments,
                           positive_class='positive')

# show performance metrics (LOGIT)
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=svm_predicted_sentiments,
                           positive_class='positive')


# COMPARE THE SVM AND LOGIT CLASSIFIERS USING Bootstrap
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# configure bootstrap
n_iterations = 100 # number of pseudo test samples
n_size = 1000      # size of each pseudo bootstrap
ids = range(1000)  # review ids in a test dataset (to use during resampling)

# accuracy for SVM
svm_accuracy = np.round( metrics.accuracy_score(test_sentiments, svm_predicted_sentiments), 3)
# accuracy for Logit
logit_accuracy = np.round( metrics.accuracy_score(test_sentiments, logit_predicted_sentiments), 3)

# Difference in performance (accuracy)
delta_test = logit_accuracy - svm_accuracy

# run bootstrap
svm_accuracy_boot = list()
logit_accuracy_boot = list()
s = 0
for i in range(n_iterations):
    
    # prepare a pseudo test set
    test_ids = resample(ids, n_samples=n_size)
    
    #actual sentiment in a pseduo test set
    actual_sentiments = test_sentiments[test_ids]
    
    #predictions in a pseudo set
    svm_predicted_sentiments = svm.predict(test_features[test_ids])
    logit_predicted_sentiments = logit.predict(test_features[test_ids])
    
    #accuracies in a pseudo set
    svm_accuracy = np.round( metrics.accuracy_score(actual_sentiments, svm_predicted_sentiments), 3)
    logit_accuracy = np.round( metrics.accuracy_score(actual_sentiments, logit_predicted_sentiments), 3)
    
    #delta for a pseudo set
    delta = logit_accuracy - svm_accuracy
    if delta > 2*delta_test:
        s = s+1
    
    pvalue = float(s)/n_iterations
    
    svm_accuracy_boot.append(svm_accuracy)
    logit_accuracy_boot.append(logit_accuracy)

# bootrsap result: p-value (high value indicates that we reject the null hypothesis (delta is die to chance))    
print 'p-value: ',pvalue
print 'Mean accuracy for SVM: ', np.round(np.mean(svm_accuracy_boot),3)
print 'Mean accuracy for Logit: ', np.round(np.mean(logit_accuracy_boot),3)

# plot the results
plt.hist(svm_accuracy_boot,bins = 30)
plt.xlabel("distribution of accuracy rates")
plt.title("Support Vector Machines")
plt.show()

plt.hist(logit_accuracy_boot,bins = 30)
plt.xlabel("distribution of accuracy rates")
plt.title("Logit Classifier")
plt.show()
