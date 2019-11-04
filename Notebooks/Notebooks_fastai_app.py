#!/usr/bin/env python
# coding: utf-8

# ### This notebook loads in the model that was trained in my 'DL_categorical-fastai' notebook. I have created a 'predictor' function which will make predictions of the number of likes that your comment will receive based on the text provided, and show examples of calling this function below.

# In[31]:


# Import libraries and then load my model. 
#Because the cuDNN libraries tend to fail to load the first time, I have the code try again on an exception error to make it work the second time.

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

    path = "models/"
    learn = load_learner(path, 'balanced_50k.pkl')
except:
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set thi

    path = "models/"
    learn = load_learner(path, 'balanced_50k.pkl')


# In[32]:


# This function will take a comment and make a prediction based on the content.

def predictor(test_comment):
 
    cat, tensor, probs = learn.predict(test_comment)
    
    #cat = np.argmax(tensor)
    
    category = str(cat)
    print(category, probs)
    if category == '0': print('This comment is predicted to be unpopular and receive no likes.') 
    else: print('This comment is predicted to be popular and receive some likes.')


# In[38]:


predictor("ASAPD")


# In[37]:


string = "What is the worth of a man these days"
predictor(string)


# In[35]:


if len(sys.argv) > 0:
    for arg in sys.argv:
        string = arg
        rec = predictor(string)
        print("\n")
        print(string)
        print(rec)
        print("\n")


# In[ ]:




