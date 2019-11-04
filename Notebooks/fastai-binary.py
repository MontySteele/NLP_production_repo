#!/usr/bin/env python
# coding: utf-8

# ###### In this notebook I attempt to fit my data using a neural network with the fastai library.

# In[3]:


# Handle imports

import pandas as pd
import numpy as np
import csv  

from sklearn.model_selection import train_test_split
import random
from datetime import datetime

from fastai.text import *

#Load the data
file_path_comments = "../Data/cleaned_comment_data_50k.csv"

clean_comments = pd.read_csv(file_path_comments, index_col = False)


# In[4]:


X = clean_comments['commentBody']
y = clean_comments['recommendations']
print(X[1])
y.head(5)


# In[5]:


len(clean_comments)


# ## In this next cell, I convert my integer target, 'recommendations', to a category target. I create bins such that comments with 0 recommendations are in bin 0 and all other comments are in bin 1.

# In[6]:


#set up bins
bin = [-1, 1, float('inf')]
label = [0, 1]
#use pd.cut function can attribute the values into its specific bins
category = pd.cut(y, bin, right = False, labels=label)
category = category.to_frame()
category.columns = ['range']
#concatenate age and its bin
df_new = pd.concat([y,category],axis = 1)
df_new['comment'] = X


# In[7]:


df_new.head(5)


# In[8]:


# Here I create a 'balanced' dataset to see if this improves my model accuracy.

df_0 = df_new.loc[df_new['range'] == 0]
df_1 = df_new.loc[df_new['range'] == 1]

print(len(df_0), len(df_1))

df_balanced = pd.concat([df_0[:(len(df_0)-1)], df_1[:(len(df_0)-1)]], axis=0)

df_balanced.head(5)


# ### Here I create a crossfold validation split. I then merge the separated comments and recommendation counts into one dataframe containing training data and one dataframe containing test data, which fastai takes as inputs.

# In[29]:


#Here I'm using the imbalanced dataset

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['comment'], df_balanced['range'], test_size=0.2, random_state=random.seed(datetime.now()))


# ### In this cell I am creating my model using data blocks created using the text classifier with the LSTM model architecture.

# In[31]:


df_nlp_data_train = pd.concat([y_train, X_train], axis=1)
df_nlp_data_test = pd.concat([y_test, X_test], axis=1)

path = '../Data/'
data_lm = TextLMDataBunch.from_df(path="", train_df=df_nlp_data_train, valid_df=df_nlp_data_test, bs=32)
data_clas  = TextClasDataBunch.from_df(path="", train_df=df_nlp_data_train, valid_df=df_nlp_data_test, bs=32)


# In[32]:


#First I train my encoder on my word corpus. This model comes with a pretrained AWD_LSTM architecture using transfer learning.

learn = language_model_learner(data_lm, AWD_LSTM)
learn.metrics=[accuracy]
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-2), moms=(0.8,0.7))

# I save my encoder
learn.save_encoder('enc')


# In[33]:


learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.3)
learn.load_encoder('enc')


# In[34]:


#Here I want to find what learning rate to use for my model.

learn.lr_find()


# In[35]:


# Set the learning rate to the point with the minimum gradient

learn.recorder.plot(suggestion=True, skip_end=15)
min_grad_lr=learn.recorder.min_grad_lr


# In[36]:


# Train my model for one cycle. Loss tends to increase with additional epochs - one cycle is sufficient.

learn.fit_one_cycle(1, min_grad_lr, moms=(0.8,0.7))


# In[37]:


# Save the model
learn.save('balanced_50k_test')
learn.export('models/balanced_50k.pkl')


# In[38]:


learn.load('balanced_50k_test')


# Here we get the predictions of the validation set, then print a sample of predictions. This is to verify that the model is not defaulting to a trivial output such as putting all predictions in the same category.

# In[39]:


preds, y_predict, loss = learn.get_preds(with_loss=True)
# get accuracy
acc = accuracy(preds, y_predict)
print('The accuracy is {0} %.'.format(acc))


#balanced: .70, imbalanced: .78
#The unbalanced model has higher accuracy


# In[40]:


# Now let's analyze the predictions on the test data

predCat = preds.data.numpy().argmax(axis=1)

print(len(preds))
print(len(y_predict))


# In[41]:


# Get the F! score, the precision and the recall

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print('F1:        {}'.format(f1_score(y_predict, predCat)))
print('Precision: {}'.format(precision_score(y_predict, predCat)))
print('Recall:    {}'.format(recall_score(y_predict, predCat)))

# Balanced: 0.74, .68, .79
# Unbalanced: .87, .78, .99

# The unbalanced model has higher precision, recall and F1


# In[44]:


# Plot my predicted values to look for overfitting

n_bins=10

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(y_test[:10000], bins=n_bins)
#axs[0].set_yscale('log')
axs[0].set_title('Actual values')
axs[1].hist(predCat[:10000], bins=n_bins)
#axs[1].set_yscale('log')
axs[1].set_title('Predictions')


# In[45]:


# Here I split out my predictions so I can create a confusion matrix

act_vs_pred = pd.concat([pd.DataFrame(y_predict.numpy()), pd.Series(predCat[:10000])], axis=1, ignore_index=True)

act_vs_pred.columns = (['actual', 'predict'])
print(act_vs_pred.head(5))

print(act_vs_pred['actual'].value_counts())
print(act_vs_pred['predict'].value_counts())

act_vs_pred.to_csv('act_vs_pred.csv', index=False)

pd.crosstab(act_vs_pred['actual'], act_vs_pred['predict'])


# ### The balanced model has a -much- lower false positive rate than the imbalanced model. Use this one!

# In[ ]:




