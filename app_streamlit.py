# Process """
from nltk import flatten
from collections import  Counter
import pandas as pd
import altair as alt

import streamlit as st
import os
from utils import *

import pandas as pd
import spacy
import nltk
import en_core_web_sm
import re 
nlp = en_core_web_sm.load() #initializing stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from collections import Counter

# stemmer = PorterStemmer()

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)
# print("Files in %r: %s" % (cwd, files))

# def remove_content(text):
#     text = re.sub(r"http\S+", "", text) #remove urls
#     text=re.sub(r'\S+\.com\S+','',text) #remove urls
#     text=re.sub(r'\@\w+','',text) #remove mentions
#     text =re.sub(r'\#\w+','',text) #remove hashtags
#     return text
# def process_text(text, stem=False): #clean text
#     text=remove_content(text)
#     text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
#     tokenized_text = word_tokenize(text) #tokenize
#     clean_text = [
#          word for word in tokenized_text
#          if (word not in stop_words and len(word)>1)
#     ]
#     if stem:
#         clean_text=[stemmer.stem(word) for word in clean_text]
#     return ' '.join(clean_text)

# def extract_ngrams(data, num):
#     n_grams = ngrams(nltk.word_tokenize(data), num)

#     n_grams_ = [grams for grams in n_grams]

#     return n_grams_


# Read pkl

df_sprm_ = pd.read_pickle('df_sprm_final.pkl')


st.title('haii')

df_sprm_['cleaned_tweets']=df_sprm_['tweet'].apply(lambda x: process_text(x))
tweet_list = df_sprm_['cleaned_tweets'].tolist()

############################
#  Keywords 1gram

one_gram =' '.join(tweet_list).split()

# remove unwanted/unimportant words in list
list_removed = ['ni', 'tak', 'yang', 'yg', 'dan', 'tu','ke','pun','apa','di','atau','sprm','ini','dia']
list1 = [ele for ele in one_gram if ele not in list_removed]

bb = Counter(list1)
#primary_count = Counter(bb)

df_1g = pd.DataFrame.from_records(bb.most_common(), columns=['1gram','Count'])
df_1g_ = df_1g.head(10)

# st.header('Primary Sector')
# primary_ =alt.Chart(df_p).mark_bar().encode(
#     y='Primary Sector',
#     x='count'
# ).properties(height=600, width= 800)

# Plot 

st.header('Most Popular Word')
primary_ =alt.Chart(df_1g_).mark_bar().encode(
    y=alt.Y('1gram',sort= '-x'),
    x='sum(Count)'
).properties(height=800, width= 1000)

st.altair_chart(primary_)



###############################################
# Bigram


# Keywords 2gram

two_gram = ''.join(tweet_list)
two_gram = extract_ngrams(two_gram,2)

cc = Counter(two_gram)

df_2g = pd.DataFrame.from_records(cc.most_common(), columns=['2gram','Count'])

list_word = df_2g['2gram']

word_2grm = []
for i in list_word:

    new_str = ' '.join(i)
    word_2grm.append(new_str)

df_2g['2gram'] = word_2grm

df_2g_ = df_2g.head(10)
print(df_2g_)

st.header('Most Popular Bigram Word')
second_ =alt.Chart(df_2g_).mark_bar().encode(
    y=alt.Y('2gram',sort= '-x'),
    x='sum(Count)'
).properties(height=800, width= 1000)

st.altair_chart(second_)


most_retweet = df_sprm_.nlargest(5, 'nretweets')

most_retweet_ = most_retweet[['username','tweet','nlikes','nretweets']]
most_retweet_.reset_index(drop=True, inplace=True)

st.table(most_retweet_)
most_retweet_list = most_retweet_['tweet'].tolist()

st.caption(f"_{most_retweet_list[0]}_")
st.caption(most_retweet_list[1])
st.caption(most_retweet_list[2])

st.title('Behaviour')

####################################################

df_sprm_['TweetPostedTime_hour'] = [d.hour for d in df_sprm_['time_']]
count =  df_sprm_['TweetPostedTime_hour'].value_counts()
print(count)
#data_ = [[0,35],[]]

df_cuba = pd.DataFrame({'Hour':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
'Frequency':[35,15,11,13,8,7,12,34,0,0,0,0,0,0,0,0,0,0,61,70,83,73,51,44]})
print(df_cuba)

cuba__=alt.Chart(df_cuba).mark_bar().encode(
    x='Hour',
    y='Frequency'
)

st.altair_chart(cuba__)

#df_hour = pd.DataFrame()

#source_ = pd.DataFrame()
#drawbarplot(x=count.values,y=count.index,xlabel='count',title='Time of the day',figsize=(10,20))









