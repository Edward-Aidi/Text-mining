#SENTIMENT ANALYSIS BASED ON LEXICON

# Imports the Google Cloud client library
from google.cloud import language   # do: pip install google.cloud
from google.cloud.language import enums
from google.cloud.language import types

#needed for text encoding issues
import six #do: pip install six
import sys #do: pip install sys
import matplotlib.pyplot as plt
import numpy as np

import os

#Setup Google  client

# DO!: CHANGE_AUTH
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/ai/Desktop/TextAnalytics/class5/text mining sentiment analysis-acedf59add73.json"
    
# Start a connection with the Google engine
client = language.LanguageServiceClient()


############### DEFINING HELPFUL FUNCTIONS ######################

# define function that returns a Google polarity score 
def sentument_scoring(movie_review_filename):    
    client = language.LanguageServiceClient()

    with open(movie_review_filename, 'r') as review_file:
        # Instantiates a plain text document.
        content = review_file.read()

    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)  
    annotations = client.analyze_sentiment(document=document)
    
    score = annotations.document_sentiment.score
    
    return(score) 

############ END OF DEFINING HELPFUL FUNCTIONS #################



################## SELECTING THRESHOLD ####################
    
# run Google engine and return a raw score for negative reviews
dir_name = '/Users/ai/Desktop/TextAnalytics/class6/reviews/train/neg/' 
score_recorder_neg = [] # keep Google sentiment scores for negative reviews
for file_name in os.listdir(dir_name):
  score_result = sentument_scoring(dir_name + file_name)
  score_recorder_neg.append(score_result)

# run Google engine and return a raw score for negative reviews
dir_name = '/Users/ai/Desktop/TextAnalytics/class6/reviews/train/pos/' 
score_recorder_pos = [] # keep Google sentiment scores for positive reviews
for file_name in os.listdir(dir_name):
  score_result = sentument_scoring(dir_name + file_name)
  score_recorder_pos.append(score_result)


#plot Google sentiment scores for reviews labeled as positive 
x = range(0, 500)
plt.bar(x, sorted(score_recorder_pos))
plt.xlabel("reviews with positive human labels")
plt.ylabel("sentiment scores")
plt.show()

#plot Google sentiment scores for reviews labeled as negative
x = range(0, 500)
plt.bar(x, sorted(score_recorder_neg))
plt.xlabel("reviews with negative human labels")
plt.ylabel("sentiment scores")
plt.show()

# define a function that takes in Google sentiment scores for reviews and a threshold value and 
# returns an accuracy rate 
def try_threshold(score_recorder_pos, score_recorder_neg, threshold_for_pos):
    binary_scores_pos = [(s>threshold_for_pos)*1 for s in score_recorder_pos]
    binary_scores_neg = [(s<=threshold_for_pos)*1 for s in score_recorder_neg]
    binary_scores = binary_scores_neg + binary_scores_pos
    accuracy = float(sum(binary_scores))/len(binary_scores)
    return(accuracy) 

############################################
#                                          #
#             COMPLETE TASKS HERE         #
#                                          #
############################################

# YOUR TASK: 
# 1) compute accuracy rates for threshold values between -1. and 1., with step = 0.05 
# 2) plot the computed accuracy rates
# 3) what threshold value maximizes the accuracy rate?

# this piece of code should be enough to get you started...
a = []
for trsh in np.arange(-1.0, 1.0, 0.05):
    a.append(try_threshold(score_recorder_pos, score_recorder_neg, trsh)) 
    #print(a)

plt.plot(a)
max(a) # the maximum accuracy value is 0.841, and the a will be at 18th position and the threshold is -1 + 17*0.05 = -0.15
a.index(max(a))

