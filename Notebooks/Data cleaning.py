#!/usr/bin/env python
# coding: utf-8

# #### Here I will document my method for cleaning my data files, which were taken from Kaggle's dataset “Comments on articles published in the New York Times” (https://www.kaggle.com/aashita/nyt-comments).
#     
# This code is broken into two sections - one cleans the data files with articles and one cleans the data files with comments.
# 
# The first block of this code combines my data files into a single large file, one each for articles and comments.

# In[2]:


import glob
import os
os.chdir("../Data")

import pandas as pd
import glob

interesting_files = glob.glob("../Comments*.csv")
df_list = []

if len(df_list) > 0: 

    for filename in sorted(interesting_files):
        df_list.append(pd.read_csv(filename))
    full_df = pd.concat(df_list)

    full_df.to_csv('allComments.csv', index=False)

interesting_files = glob.glob("Articles*.csv")
df_list = []

if len(df_list) > 0: 
    for filename in sorted(interesting_files):
        df_list.append(pd.read_csv(filename))
    full_df = pd.concat(df_list)

    full_df.to_csv('allArticles.csv', index=False)


# # Initial goals: 
# 
# -Make sure the contents of each field are the correct type and have no missing data (i.e. scrub the 'NaN' from the 'abstract' field)
# 
# -Make sure that the data comes properly tokenized
# 
# -Convert all words to lowercase (to avoid confusion between uppercase and lowercase versions of the same word)

# ###### From the above code, the only integer columns are 2, 7 and 9. The rest are string columns and need to be converted to lowercase. 

# ###### I want to check for cells that have missing elements.

# ###### Now we want a separate section to clean the comment files.

# In[7]:


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



#For all data - train = pd.read_csv("/root/Springboard/Data/cleaning/allComments.csv")

train = pd.read_csv("allComments.csv")

#train['recommendations'].head(5)
train


# In[8]:


train= train.astype(str)
train.fillna(0)
strings = [1, 5, 10, 22, 24, 25, 26, 29, 30, 33]

wn = nltk.WordNetLemmatizer()

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


# In[9]:


print("Any null values left: "), print(clean_comments.isnull().values.any())


# In[10]:


print(len(clean_comments))
clean_comments.nunique()


# I see that there are no null values remaining, but looking at the dataframe I see that several columns contain nothing but 'nan' strings or otherwise have only one value. I want to drop the commentTitle (contains only <br/> or nan), recommendedFlag, reportAbuseFlag, status, timespeople, userTitle and userURL columns.

# In[11]:


clean_comments.drop(columns=['commentTitle', 'recommendedFlag', 'reportAbuseFlag', 'status', 'timespeople', 'userTitle', 'userURL'], axis=1, inplace=True)

clean_comments.head(5)


# ###### I check my working directory to make sure I am saving the files where I want them stored.

# In[12]:


com_file_name = "cleaned_comment_data.csv"
clean_com_csv = clean_comments.to_csv(com_file_name, encoding='utf-8', index=False)


# In[14]:


nbconvert to script "Data cleaning.py"


# In[ ]:




