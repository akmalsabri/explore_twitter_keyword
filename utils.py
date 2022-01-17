import twint
import spacy
import nltk
import en_core_web_sm
import re 
nlp = en_core_web_sm.load() #initializing stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def keywords_explore(keywords_ , start_, end_, limit_ ):

    '''
    keywords - str
    
    '''

    c = twint.Config()

    # keywords
    c.Search = keywords_
    c.Store_object = True
    #config.Lang = "en"
    
    # variables
    c.Limit = limit_
    c.Since = start_
    c.Until = end_

    # store as csv
    c.Store_csv = True

    # format of csv
    c.Custom = ["date", "time", "username", "tweet", "link", "likes", "retweets", "replies", "mentions", "hashtags"]

    name_ = keywords_.lower()
    print(name_)
    c.Output = "{}_.csv".format(name_)
    
    
    #running search
    twint.run.Search(c)


def keywords_df(keywords_ , start_, end_, limit_ ):
    '''
    Input
    ----------
    keywords_ - str
    start_ - str [yy-mm-dd]
    end_ - str [yy-mm-dd]
    limit_ - int()

    Output
    --------
    df_ - dataframe
    '''

    c = twint.Config()
    c.Search = keywords_
    c.Store_object = True
    c.Pandas = True

    c.Limit = limit_
    c.Since = start_
    c.Until = end_



    twint.run.Search(c)
    df_output = twint.storage.panda.Tweets_df

    #c.Custom = ["date", "time", "username", "tweet", "link", "likes", "retweets", "replies", "mentions", "hashtags"]


    return df_output


def remove_content(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text=re.sub(r'\S+\.com\S+','',text) #remove urls
    text=re.sub(r'\@\w+','',text) #remove mentions
    text =re.sub(r'\#\w+','',text) #remove hashtags
    return text
def process_text(text, stem=False): #clean text
    text=remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
    tokenized_text = word_tokenize(text) #tokenize
    clean_text = [
         word for word in tokenized_text
         if (word not in stop_words and len(word)>1)
    ]
    if stem:
        clean_text=[stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)

def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)

    n_grams_ = [grams for grams in n_grams]

    return n_grams_



