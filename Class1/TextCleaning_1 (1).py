import nltk 
nltk.download('gutenberg') #package to work with a dataset of free-access books
nltk.download('punkt')
from nltk.corpus import gutenberg 
from pprint import pprint #pretty printing

#texts
alice = gutenberg.raw(fileids='carroll-alice.txt') 
pprint(len(alice[0:8])) #print first 8 characters in the corpus
 

#SENTENCE TOKENIZATION
default_st = nltk.sent_tokenize #define sentence tokenization function
alice_sentences = default_st(text=alice) 
print '\nTotal sentences in alice:', len(alice_sentences) 
print 'First 5 sentences in alice:-', pprint(alice_sentences[0:5])


#WORD TOKENIZATION
sentence = "The brown fox wasn't that quick and he couldn't win the race"

default_wt = nltk.word_tokenize
words = default_wt(sentence)
print words   

# tokenizarion by punctuation rules
wordpunkt_wt = nltk.WordPunctTokenizer()
words = wordpunkt_wt.tokenize(sentence)
print words

# tokenization is done using white space
whitespace_wt = nltk.WhitespaceTokenizer()
words = whitespace_wt.tokenize(sentence)
print words

#TEXT NORMALIZATION
import re 
import string  
corpus = ["The brown fox wasn't that quick and he couldn't win the race", 
          "Python is an amazing language !@@"]

#define a tokenization function
def tokenize_text(text): 
    sentences = nltk.sent_tokenize(text) 
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences] 
    return word_tokens

token_list = [tokenize_text(text) for text in corpus]
pprint(token_list) #print tokens

#define function for removing special characters
def remove_characters_after_tokenization(tokens): 
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation))) 
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) 
    return filtered_tokens

filtered_list_1 =  [filter(None,[remove_characters_after_tokenization(tokens)
    for tokens in sentence_tokens]) for sentence_tokens in token_list]
print filtered_list_1 #print clean tokens


# open and run contractions.py first and then do the lines below
def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
    
expanded_corpus = [expand_contractions(sentence, CONTRACTIONS_MAP) for sentence in corpus]
print expanded_corpus
