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


# # Initial goals: 
# 
# -Make sure the contents of each field are the correct type and have no missing data (i.e. scrub the 'NaN' from the 'abstract' field)
# 
# -Make sure that the data comes properly tokenized
# 
# -Convert all words to lowercase (to avoid confusion between uppercase and lowercase versions of the same word)
# 
# Several of these data columns (articleID, articleWordCount, multimedia, printPage) contain only integers or single lowercase words.

# In[ ]:


#nltk.download('wordnet')
#for column in train:
    #print(train[column].get_dtype_counts())
    
print(train.dtypes.value_counts())

print("")

for column in train:
    print(train[column].dtypes)


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


# ###### I want to check for cells that have missing elements.

# In[ ]:


print("Any null values left: ")
print(clean_art.isnull().values.any())
print("Dataset size: ")
print(len(train))


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


print("Any null values left: "), print(clean_comments.isnull().values.any())


# In[ ]:


print(len(clean_comments))
clean_comments.nunique()


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


# ###### Here I have saved the cleaned data files to storage. This is a good break point where I can use these files to begin looking at possible ML models. 
#     
#    In the next section, I will explore the data, looking at features and ways to visualize the feature set. I begin by re-importing the libraries that I will use. That way, this section of the notebook can be reran separately later. I will load the cleaned files in so that I know I am not making any changes to the stored version.

# In[ ]:


import pandas as pd
import numpy as np
import nltk
#ntlk.download()
sent_token = nltk.sent_tokenize
import csv  
from nltk import sent_tokenize, word_tokenize, pos_tag
import re
from sklearn.feature_extraction.text import CountVectorizer
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

file_path_art = r"C:\Users\msteele9\Documents\Springboard\Springboard\Data\cleaned_article_data.csv"
clean_art = pd.read_csv(file_path_art, index_col = False)
clean_art

file_path_comments = r"C:\Users\msteele9\Documents\Springboard\Springboard\Data\cleaned_comment_data.csv"
clean_comments = pd.read_csv(file_path_comments, index_col = False)


# ###### In the above cell, I loaded in the cleaned article data. Here I load in the cleaned data and do preprocessing to run the data through a random forest model. 
#     
#    I want to make sure that my data can actually be loaded and that my simple test will produce results of some sort - this way I know that my data will not break when I try to load it.
#    
#    My target is the number of recommendations that the comment receives. I want to know how well a simple model can predict the number of recommendations a comment will receive when given the rest of that comment's data.
# 

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

features = clean_comments.columns.tolist()
print(features)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

features = clean_comments.columns.tolist()
output = 'recommendations'
features.remove('recommendations')

for column in clean_comments.columns:
    clean_comments[column] = clean_comments[column].astype(str)
    if clean_comments[column].dtype == type(object):
        clean_comments[column] = le.fit_transform(clean_comments[column])

#print(features)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

clean_comments.astype(float)
clean_comments.head(5)

#clean_comments.dtypes


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

param = {'n_estimators': [5, 10, 20],
         'max_depth': [5, 10, 20] }

gs=GridSearchCV(rf, param, cv=5, n_jobs=5)

gs_fit = gs.fit(clean_comments[features], clean_comments[output])


# In[ ]:


pd.DataFrame(gs_fit.cv_results_).sort_values(by=['rank_test_score']).head(5)


# ###### A quick GridSearch with a random forest classifier was able run on my data set and produce an accuracy score. My goal was to confirm that I can run ML algorithms on my data set and get sensible results; this test seems to verify this.
# 
# Here I am looking to see what the best model is doing. Only a few of our features have a strong impact on the number of recommendations. It looks like the dominant features are the approve date and the comment body. These intuitively make sense - we expect the content of the comment to be one of if not the most important feature, and the time the comment is posted likely determines how many people will see it.

# In[ ]:


import matplotlib.pyplot as plt

importances = gs_fit.best_estimator_.feature_importances_

std = np.std([tree.feature_importances_ for tree in gs_fit.best_estimator_.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(clean_comments[features].shape[1]):
    print("%d. feature %d : (%s) (%f)" % (f + 1, indices[f], features[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(clean_comments[features].shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(clean_comments[features].shape[1]), indices)
plt.xlim([-1, clean_comments[features].shape[1]])
plt.show()


# In[ ]:





# In[ ]:




