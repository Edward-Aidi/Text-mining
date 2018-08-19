
#IN TERMINAL: pip install tweepy
#IN TERMINAL: pip install twitter
#IN TERMINAL: pip install jsonpickle

# Import the necessary methods from tweepy library

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import jsonpickle 


CONSUMER_KEY = 'I3jWt5UvA9gXyEUf1Md3Jv8MP'
CONSUMER_SECRET = 'oTpfKSYEQ0vwg4lckVoN26seeIXpW8oE5BhN6dWYwJbyUg1foP' 
ACCESS_TOKEN = '908526645182595073-b3qxfqo1G2Gb1tGmTLug4shZAwzdgEX' 
ACCESS_TOKEN_SECRET = 'KpMIMOEkqqkdDvxRQWYuX7Kd6ke4AD1At5ZILf99xIrey' 


# Basic listener that just prints received tweets (STREAMING CURRENT TWEETS)
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)
        
if  __name__ == '__main__':

    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    stream = Stream(auth, l)
    
    # This line filter Twitter Streams to capture data by the keywords
    stream.filter(languages=["en"], track=['Pepsi'])
    
    
# You can run this sript IN THE TERMINAL by typing: python /Users/nevskaya/Dropbox/TextAnalytics/Codes/Yulia_twitter.py
# INTERRUPT by: Ctrl+C
# DIRECT OUTPUT TO A FILE (use Treminal): python /Users/nevskaya/Dropbox/TextAnalytics/Codes/Yulia_twitter.py > twitter_data_YN.txt    

