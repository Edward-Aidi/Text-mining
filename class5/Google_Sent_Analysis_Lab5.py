# Imports the Google Cloud client library
from google.cloud import language   # do: pip install google.cloud
from google.cloud.language import enums
from google.cloud.language import types

#needed for text encoding issues
import six #do: pip install six
import sys #do: pip install sys

import os

# CHANGE_AUTH
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/ai/Desktop/TextAnalytics/class5/text mining sentiment analysis-acedf59add73.json"
    
# Start a connection with the Google engine
client = language.LanguageServiceClient()


#SENTIMENT ANALYSIS BASED ON LEXICON

#define a printing function
def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0


# define a function to open a text file and use the Google engine on it    
def analyze(movie_review_filename):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    with open(movie_review_filename, 'r') as review_file:
        # Instantiates a plain text document.
        content = review_file.read()
        print(content)

    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)  
    annotations = client.analyze_sentiment(document=document)
    
    print_result(annotations)


# directory with data
# Note: folder "neg" contains negative reviews, folder "pos" contains positive ones    
# DO: change the path for your machine 
dir_name = '/Users/ai/Desktop/TextAnalytics/class5/reviews/train/neg/'

# see the text and do analysis for 1 movie review
file_name = dir_name + '12499_2.txt'
analyze(file_name)

# see the text and do analysis for 5 movie reviews
for file_name in os.listdir(dir_name)[0:5]:
    analyze(dir_name + file_name)
 

#IDENTIFYING ENTITIES in movie reviews

# salience: between 0 and 1.0
# salience shows the importance of an entity to the entire document

# text to be analyzed
text = """The movie is very long , 
I believe it could have cut to 1/2 without causing any problems to the story. 
Its the type of movie you can see in a boring night which you want to get bored more ! 
Ashton Kutcher was very good . Kevin Costner is OK."""

# function 
def entities_text(text):
    """Detects entities in the text."""
    client = language.LanguageServiceClient()

    # checking for the correct encoding
    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Instantiates a plain text document.
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects entities in the document. You can also analyze HTML with:
    #   document.type == enums.Document.Type.HTML
    entities = client.analyze_entities(document).entities

    # entity types from enums.Entity.Type
    entity_type = ('UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
                   'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER')

    for entity in entities:
        print('=' * 20)
        print(u'{:<16}: {}'.format('name', entity.name))
        print(u'{:<16}: {}'.format('type', entity_type[entity.type]))
        print(u'{:<16}: {}'.format('metadata', entity.metadata))
        print(u'{:<16}: {}'.format('salience', entity.salience))
        print(u'{:<16}: {}'.format('wikipedia_url',
              entity.metadata.get('wikipedia_url', '-')))

# example 1       
entities_text("I love iPhone") 
 
# example 2      
entities_text(text)



# SENTIMENT ANALYSIS with ENTITY IDENTIFICATION
  
def entity_sentiment_text(text):
    """Detects entity sentiment in the provided text."""
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    # Detect and send native Python encoding to receive correct word offsets.
    encoding = enums.EncodingType.UTF32
    if sys.maxunicode == 65535:
        encoding = enums.EncodingType.UTF16

    result = client.analyze_entity_sentiment(document, encoding)

    for entity in result.entities:
        print('Mentions: ')
        print(u'Name: "{}"'.format(entity.name))
        for mention in entity.mentions:
            print(u'  Begin Offset : {}'.format(mention.text.begin_offset))
            print(u'  Content : {}'.format(mention.text.content))
            print(u'  Magnitude : {}'.format(mention.sentiment.magnitude))
            print(u'  Sentiment : {}'.format(mention.sentiment.score))
            print(u'  Type : {}'.format(mention.type))
        print(u'Salience: {}'.format(entity.salience))
        print(u'Sentiment: {}\n'.format(entity.sentiment))


entity_sentiment_text(text)



# MEASURING ACCURACY OF GOOGLE'S SENTIMENT ANALYSIS

# define function that returns a binary indicator for a text polarity (0 - negative, 1 - positive)  
def sentument_binary(movie_review_filename):    
    client = language.LanguageServiceClient()

    with open(movie_review_filename, 'r') as review_file:
        # Instantiates a plain text document.
        content = review_file.read()

    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)  
    annotations = client.analyze_sentiment(document=document)
    
    score = annotations.document_sentiment.score
    binary_indicator = 1*(score>0)
    
    return(binary_indicator) 


# computing accuracy of Google algorithm for negative movie reviews
# DO: change the path for your machine!    
dir_name = '/Users/ai/Desktop/TextAnalytics/class5/reviews/train/neg/' 

#number of reviews (all are negative)
N_reviews = len(os.listdir(dir_name))
print('Number of negative reviews: ', N_reviews)

#keep records of binary indicators
review_recorder = [] 

# run Google engine
for file_name in os.listdir(dir_name):
  binary_result = sentument_binary(dir_name + file_name)
  review_recorder.append(binary_result)

#number of reviews identified by Google as positive
N_pos_Google = sum(review_recorder)

#ANSWER: accuracy rate for negative reviews
acuracy_neg = 100.*(N_reviews - N_pos_Google)/N_reviews
print 'Accuracy for negative reviews: ', acuracy_neg,'%'


###############################################################
####    Measure Google's accuracy for positive reviews here ###
###############################################################



dir_name = '/Users/ai/Desktop/TextAnalytics/class5/reviews/train/pos/' 

#number of reviews (all are negative)
N_reviews = len(os.listdir(dir_name))
print('Number of negative reviews: ', N_reviews)

#keep records of binary indicators
review_recorder = [] 

# run Google engine
for file_name in os.listdir(dir_name):
  binary_result = sentument_binary(dir_name + file_name)
  review_recorder.append(binary_result)

#number of reviews identified by Google as positive
N_pos_Google = sum(review_recorder)

#ANSWER: accuracy rate for negative reviews
acuracy_pos = 100.* N_pos_Google/N_reviews
print 'Accuracy for positive reviews: ', acuracy_pos,'%'





