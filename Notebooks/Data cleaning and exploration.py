#!/usr/bin/env python
# coding: utf-8

# #### Here I will document my method for cleaning my data files, which were taken from Kaggle's dataset “Comments on articles published in the New York Times” (https://www.kaggle.com/aashita/nyt-comments).
#     
# This code is broken into two sections - one cleans the data files with articles and one cleans the data files with comments.
# 
# The first block of this code combines my data files into a single large file, one each for articles and comments.

# In[ ]:


import glob
import os
os.chdir(r"C:\Users\msteele9\Documents\Springboard\prod_repo\NLP_production_repo\Data")

import pandas as pd
import glob
interesting_files = glob.glob("Articles*.csv")
df_list = []
for filename in sorted(interesting_files):
    df_list.append(pd.read_csv(filename))
full_df = pd.concat(df_list)

full_df.to_csv('allArticles.csv', index=False)


# # Disabled - make sure comment data is available before re-enabling
# 
# interesting_files = glob.glob("../../Comments*.csv")
# df_list = []
# for filename in sorted(interesting_files):
#     df_list.append(pd.read_csv(filename))
# full_df = pd.concat(df_list)
# 
# full_df.to_csv('allComments.csv', index=False)

# In[ ]:


import pandas as pd
import numpy as np
import nltk
#nltk.download('wordnet')
import re
sent_token = nltk.sent_tokenize
import csv  
from nltk import sent_tokenize, word_tokenize, pos_tag
import re
from sklearn.feature_extraction.text import CountVectorizer
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

#For all data - train = pd.read_csv("/root/Springboard/Data/cleaning/allArticles.csv")
train = pd.read_csv(r"C:\Users\msteele9\Documents\Springboard\prod_repo\NLP_production_repo\Data\allArticles.csv", index_col = False)
#train = pd.read_csv(r"/mnt/c/Users/msteele9/Documents/Springboard/Springboard/Data/allArticles.csv")

train.head(5)


# In[ ]:





# ###### From the above code, the only integer columns are 2, 7 and 9. The rest are string columns and need to be converted to lowercase. 

# In[ ]:


sampleSize = 5000
train = pd.read_csv(r"C:\Users\msteele9\Documents\Springboard\prod_repo\NLP_production_repo\Data\allArticles.csv", header=0, nrows=sampleSize)
#train.head(5)

wn = nltk.WordNetLemmatizer()

nonstrings = [2, 7, 9]


train= train.astype(str)
train.fillna(0)

def clean_articles(doc):
    for index, column in enumerate(doc):
        if index in nonstrings:
            doc[column] = doc[column].astype(str)
            continue
        doc[column] = doc[column].str.replace('[^\w\s]','')
        doc[column] = doc[column].str.lower()
        #doc[column] = doc[column].str.strip()
        doc[column] = doc[column].replace(np.nan, ' ', regex=True)
        doc[column].apply(nltk.word_tokenize)
        doc[column].apply(lemmatize_text)
        doc[column] = [token for token in doc[column] if token not in stop_words]     
    return doc

def lemmatize_text(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

clean_art = clean_articles(train)

clean_art.head(5)

#print('\n'.join(clean))


# In[ ]:





# ###### Now we want a separate section to clean the comment files.

# In[ ]:


#For all data - train = pd.read_csv("/root/Springboard/Data/cleaning/allComments.csv")

sampleSize = 5000
train = pd.read_csv(r"C:\Users\msteele9\Documents\Springboard\prod_repo\NLP_production_repo\Data\allComments.csv", nrows=sampleSize)

#train['recommendations'].head(5)
train


# In[ ]:


train= train.astype(str)
train.fillna(0)
strings = [1, 5, 10, 22, 24, 25, 26, 29, 30, 33]

def lemmatize_text(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

def clean_articles(doc):
    for index, column in enumerate(doc):
        if index in strings:         
            doc[column] = doc[column].str.replace('[^\w\s]','')
            doc[column] = doc[column].str.lower()
            #doc[column] = doc[column].str.strip()
            doc[column] = doc[column].replace(np.nan, '', regex=True)
            doc[column].apply(nltk.word_tokenize)
            doc[column].apply(lemmatize_text)
            doc[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        else:
            doc[column] = doc[column].astype(str)
            continue
    return doc

clean_comments = clean_articles(train)
#The second command takes awhile to run
clean_comments.head(5)

clean_comments['recommendations'].head(50)


# In[ ]:





# In[ ]:





# I see that there are no null values remaining, but looking at the dataframe I see that several columns contain nothing but 'nan' strings or otherwise have only one value. I want to drop the commentTitle (contains only <br/> or nan), recommendedFlag, reportAbuseFlag, status, timespeople, userTitle and userURL columns.

# In[ ]:


clean_comments.drop(columns=['commentTitle', 'recommendedFlag', 'reportAbuseFlag', 'status', 'timespeople', 'userTitle', 'userURL'], axis=1, inplace=True)

clean_comments.head(5)


# ###### I check my working directory to make sure I am saving the files where I want them stored.

# In[ ]:


# 
import os

os.chdir(r"C:\Users\msteele9\Documents\Springboard\prod_repo\NLP_production_repo\Data")
art_file_name = "cleaned_article_data.csv"
clean_art_csv = clean_art.to_csv(art_file_name, encoding='utf-8', index=False)


# In[ ]:


com_file_name = "cleaned_comment_data.csv"
clean_com_csv = clean_comments.to_csv(com_file_name, encoding='utf-8', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




