import json

#small example
jsonData = '{"name": "Frank", "age": 39}'
jsonToPython = json.loads(jsonData)
print jsonToPython
print jsonToPython["name"]

#OPENING YELP REVIEWS DATASETS
#read the data from disk and split into lines
#we use .strip() to remove the final (empty) line
with open("/Users/nevskaya/Dropbox/TextAnalytics/DATASETS/Yelp_Reviews/yelp_academic_dataset_review.json") as f:
    reviews = f.read().strip().split("\n")
 
#each line of the file is a separate JSON object
reviews = [json.loads(review) for review in reviews] 
print reviews[2]
 
#we're interested in the text of each review and the stars rating, so we load these into 
# separate lists
texts = [review['text'] for review in reviews]
stars = [review['stars'] for review in reviews]

print texts[2]
print stars[2]


