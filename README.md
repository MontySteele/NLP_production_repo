### Springboard Capstone project proposal ###

This project was submitted at the end of my course with Springboard's Machine Learning track. The ultimate goal of this project is to build a web api in which a user can enter a sample comment for a news article and receive a prediction of how many 'likes' or 'recommendations' their comment will receive. This program could be useful to web managers to create a spam filter that automatically removes low-value spam comments or places comments that are predicted to eb popular at the top of a discussion thread for easy viewing. 

This repo builds and saves the trained neural network which can be used for this task, as well as providing a sample app to test the model with your own comments from the command line.

### Data for this project ###

The data for this project is saved inder the /Data directory as .csv files. One group of files contains data on ~9000 New York Times articles. The other group of files  contains ~2M comments made on these articles. The comment data files were stored as git LFS objects, but due to these files exceeding the maximum size permitted by LFS storage, they are not currently available.

This data was taken from the ["New York Times Comments"](https://www.kaggle.com/aashita/nyt-comments "New York Times Comments") dataset on Kaggle.

### Dependencies ###

This project uses fastai to create an LSTM model and apply transfer learning. It is recommended that fastai be installed under a virtual environment. To create a virtual environment using conda (anaconda) with all necessary dependencies installed, run the following command:

conda create -n fastai python=3.6

Documentation for installing conda can be found [here.](https://docs.anaconda.com/anaconda/install/)

### Using this project ###

After cloning this repo, make sure that the comment data files are places in the Data directory. Then run the following commands from the Notebook directory:

python "Data cleaning and exploration.py"
python "fastai-binary.py"
python "Notebooks-fastai-app.py" "THIS IS A COMMENT!"

You can replace the last argument to the previous command with any comment of your choice. You will receive a message such as: "This comment was placed in category 0. This means that we predict your comment will have between -1 and 1 recommendations."

The "Data cleaning and exploration.py" program will output cleaned data files which are fed into "fastai-binary.py" to train the model. The model is saved as "trained_model.pkl" in the Notebooks folder. The model is not available on this repo as it exceeds the file size limits (~143 MB).
