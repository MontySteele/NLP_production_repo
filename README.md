## Springboard Capstone project proposal ###

This project was submitted at the end of my course with Springboard's Machine Learning track. The ultimate goal of this project is to build a web api in which a user can enter a sample comment for a news article and receive a prediction of how many 'likes' or 'recommendations' their comment will receive. This program could be useful to web managers to create a spam filter that automatically removes low-value spam comments or places comments that are predicted to eb popular at the top of a discussion thread for easy viewing. 

This repo builds and saves the trained neural network which can be used for this task, as well as providing a sample app to test the model with your own comments from the command line and an API which can provide the same functionality remotely via a webpage or through direct API calls for processing many comments.

### Data for this project ###

The data for this project is saved under the /Data directory as .csv files. One group of files contains data on ~9000 New York Times articles. The other group of files  contains ~2M comments made on these articles. The comment data files were stored as git LFS objects, but due to these files exceeding the maximum size permitted by LFS storage, they are not currently available.

This data was taken from the ["New York Times Comments"](https://www.kaggle.com/aashita/nyt-comments "New York Times Comments") dataset on Kaggle.

Because the raw data files exceed the size limits of Github's free storage, I have provided only a small sample (5k comments) in the /Data directory here.

### Dependencies ###

This project uses fastai to create an LSTM model and apply transfer learning. It is recommended that fastai be installed under a virtual environment. To create a virtual environment using conda (anaconda) with all necessary dependencies installed, run the following command:

conda create -n fastai python=3.6

Documentation for installing conda can be found [here.](https://docs.anaconda.com/anaconda/install/)

### Hardware ###

I trained this model on a Google Cloud server using 8 vCPU cores and one Nvidia k80 GPU for ~6 hours at a cost of ~$10. The fastai library is an API that sits on top of pytorch. Pytorch is designed to work with or without a GPU; however, it is ill-advised to attempt to train this model on a CPU alone. Once the model is trained, it can be run very quickly on new comments with only a CPU, though a GPU will result in a significant speedup for large batch processing.

### Notebooks ###

I documented my work in three Jupyter Notebooks located in the /Notebooks folder. 

First, "Data cleaning.ipynb" shows how the raw data files from Kaggle are loaded and cleaned.

Second, "fastai-binary.ipynb" shows how my AWD-LSTM Neural Network is trained and tested.

Third, I provide an example of how to use this model in "Notebooks-fastai-app.py". The predictor function used here is the same as in the API.

### Setting up this project ###

After cloning this repo, make sure that the comment data files are places in the Data directory. Then run the following commands from the Notebook directory:

python "Data cleaning.py"

python "fastai-binary.py"

python "Notebooks-fastai-app.py" "THIS IS A COMMENT!"

You can replace the last argument to the previous command with any comment of your choice. You will receive a message such as: "This comment is predicted to be unpopular and receive no likes."

The "Data cleaning and exploration.py" program will output cleaned data files which are fed into "fastai-binary.py" to train the model. The model is saved as "balanced_50k.pkl" in the Notebooks/models folder. A pre--trained model is not available on this repo as it exceeds the file size limits (~143 MB).

### Running this project in a production environment ###

After cloning this repo, make sure that the comment data files are places in the Data directory. Then run the following commands from the Notebook directory (Data cleaning may be skipped if it was ran in the previous section):

python "Data cleaning and exploration.py"

python "fastai-binary.py"

python api.py

The api will now be listening for incoming requests at port 8000. To reach the webserver, go to:

http://<Your_IP_address>:8000

To sent an api call, send an html request using the following format: http://<Your_IP_address>:8000/comment/api/?comment=<YOUR_COMMENT_HERE>

The API will return a JSON dictionary formatted as: { <YOUR_COMMENT_HERE> : <0 or 1> }

Comments that are predicted to receive no likes will return a 0 value dictionary entry. Comments that are predicted to receive some likes will return a 1 instead.

### Model Architecture ###

In my production model, I am using an AWD-LSTM (ASDG Weight-Dropped Long Short-term Memory) architecture implemented by the fastai library. AWD-LSTM architectures are a type of Recurrent Neural Network with additional types of dropouts. In addition to the weight dropout common in LSTM, dropout is also used in the embedding layer, the embedding matrix and the output of hidden layers. The documentation for this implementation is available [here.](https://docs.fast.ai/text.models.html)

RNN models are seen as well-suited to NLP analysis because of their 'memory' of recent data that lets them find, e.g. patterns in sentence structure. The AWD-LSTM model is a state-of-the-art improvement on the basic RNN that is well-suited for comment classification.

The model is trained in two steps- first I train a language model which tried to learn our comment dataset by predicting the next word in a given sentence. In this step I apply transfer learning by borrowing the weights of a language model pretrained in the WikiText-103 dataset.

Once the language model is trained on my word corpus, I then train it for classification tasks by giving it a training dataset and asking for predictions of the comment class ('popular' or 'unpolular'). This model is then measured against a test dataset. Both the train and test dataset are balanced to contain equal numbers of 'popular' and 'unpopular' comments.

### Results summary ###

My model has a final accuracy of ~70% when run on the full dataset. The f1 score is 0.73, and a confusion matrix is given below (values rounded):

          | Predicted unpopular  | Predicted popular|
          | -------------        | -------------    |
Unpopular |      0.253           |     0.246        |
Popular   |      0.072           |     0.429        |

The model is much better at successfully predicting 'popular' comments than 'unpolular' comments, with only slightly better than even success on 'unpopular' comments. 

It is worth keeping in mind that 'unpopular' comments are not necessarily spam, but also grammatically and contextually valid comments that are simply uninteresting to the target audience. On a small experiment, the model was able to successfully filter out as 'unpopular' 25 of 25 cases of 'spam' generated by entering random strings of letters.

### Other attempted models ###

Early on in this project, I attempted to develop a regression model which could predict the exact number of recommendations the comment would get. I started by trying a logistic regression over the comment metadata, but I found that the model would always try to predict the most common class; that is, the model couldn't actually learn anything from the data. In this dataset, this would give an accuracy of ~20%.

I then tried using neural networks because those are widely seen as the state-of-the-art approach for NLP-based problems. I worked with Keras and Tensorflow to design my own neural network, which was able to achieve ~33% accuracy on the regression problem (that is, identifying the exact number of comments 33% of the time). The neural network was able to improve over always guessing the most common class. However, I found that my model's 'like' distribution was consistently smaller than the sample distribution; that is, it would consistently underestimate the correct number of 'likes' for a comment.

I then modified my approach by converting my problem from regression to classification; instead of predicting the number of likes, I sorted my comments into 'unpopular' comments (0 likes) or 'popular' comments (1 or more likes). I also switched to using the LSTM (Long Short-term Memory) architecture, a popular modification of Recurrent Neural Networks (RNN's) that are better able to handle dependencies between parts of the data, such as relationships between words in a sentence. This approach scored ~58% accuracy on the test data.

At this point I switched to using pyTorch and fastai as the fastai library offers a quick and easy implementation of an AWD-LSTM model with transfer learning applied, as described above. Switching to this approach increased my accuracy as high as 86%, but I found that the output prediction distribution contained very few 'unpopular' comments; that is, the model would predict almost all comments as 'popular', so the high accuracy score was an artifact of the test set consisting of relatively few 'unpopular' comments. The false-negative rate was very high, with accuracy on 'unpolular' test comments as low as ~25%.

To fix this problem, I used a balanced training and testing data set where the number of 'popular' and 'unpopular' comments were equal. I retrained my model on this set and my overall accuracy was reduced to ~70%, but the model's false-negative rate went down significantly and the model's accuracy on 'unpolular' comments increased to 51%. My model still performs poorly on these comments but is better than random.


### Final thoughts ###

This model could be improved by incorporating the comment metadata into the final analysis; for example, the title or text of the original article the comment was writtenin response to. I think that this would allow for a more fine-grained analysis of the input comment's context and would improve the model's accuracy, especially for 'unpopular' comments. A comment that might receive no 'likes' in one context may receive them in a different context.

I think that this model does succeed in providing useful information when working with a generally difficult problem, so while the model's overall accuracy may not be stellar, I think that that is partly due to the fundamentally difficult problem of determining the popularity of the comment rather than simply filtering out spam.
