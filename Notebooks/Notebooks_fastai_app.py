#!/usr/bin/env python
# coding: utf-8

# ### This notebook loads in the model that was trained in my 'DL_categorical-fastai' notebook. I have created a 'predictor' function which will make predictions of the number of likes that your comment will receive based on the text provided, and show examples of calling this function below.

# In[1]:


import os
cwd = os.getcwd()
print(cwd)


# In[4]:


from fastai.text import *
from fastai.basics import *
try:    
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set thi

    path = r'C:\Users\msteele9\Documents\Springboard\Springboard\Notebooks'
    learn = load_learner(path, 'trained_model.pkl')
except:
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set thi

    path = "models/"
    learn = load_learner(path, 'fit_lstm_binary_5k.pth')


# In[5]:


def predictor(test_comment):
    
    bin = [-1, 1, 5, 1000000]
    
    cat, tensor, probs = learn.predict(test_comment)
    print(probs)
    
    category = str(cat)
    leftBin = str(bin[int(str(cat))])
    rightBin = str(bin[int(str(cat))+1])
    
    print('This comment was placed in category ' + category + '. This means that we predict your comment will have between ' + leftBin + ' and ' + rightBin + ' recommendations.')


# In[6]:


try:
    predictor("This is a test of the emergency comment system that was placed in category 2 ")
except:
    predictor("This is a test of the emergency comment system that was placed in category 2 ")


# In[7]:


string = "Category 0"
predictor(string)


# In[8]:


if len(sys.argv) > 0:
    for arg in sys.argv:
        string = arg
        rec = predictor(string)
        print("\n")
        print(string)
        print(rec)
        print("\n")


# In[ ]:




