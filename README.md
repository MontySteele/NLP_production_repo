### Springboard Capstone project proposal ###

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
