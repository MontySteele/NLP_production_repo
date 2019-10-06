### Capstone project proposal ###

A project using NLP on the dataset “Comments on articles published in the New York Times” (https://www.kaggle.com/aashita/nyt-comments). 

The ultimate goal of this project is to build a web interface in which a user can enter a sample comment for a news article and receive a prediction of how many 'likes' or 'recommendations' their comment will receive. This program could be useful to authors to create a spam filter that removes low-value comments.


### Data for this project ###

The data for this project is saved inder the /Data directory as .csv files. One group of files contains data on ~9000 New York Times articles. The other group of files  contains ~2M comments made on these articles. The comment data files were stored as git LFS objects, but due to these files exceeding the maximum size permitted by LFS storage, they are not currently available.

This data was taken from the ["New York Times Comments"](https://www.kaggle.com/aashita/nyt-comments "New York Times Comments") dataset on Kaggle.

### Using this project ###

After cloning this repo, make sure that the comment data files are places in the Data directory. Then run the following commands from the Notebook directory:

python "Data cleaning and exploration.py"
python "fastai-binary.py"
python "Notebooks-fastai-app.py" "THIS IS A COMMENT!"

You can replace the last argument to the previous command with any comment of your choice. You will receive a message such as: "This comment was placed in category 0. This means that we predict your comment will have between -1 and 1 recommendations."



