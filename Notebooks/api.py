from flask import Flask, render_template, request
app = Flask(__name__)

from fastai.text import *
from fastai.basics import *
import json

path = r'/home/montgomerysteele/nlp_data_msteele/Notebooks'
learn = load_learner(path, 'trained_model.pkl')

def predictor(test_comment):
        
        bin = [-1, 1, 5, 1000000]
            
        cat, tensor, probs = learn.predict(test_comment)           

        category = str(cat)
        leftBin = str(bin[int(str(cat))])
        rightBin = str(bin[int(str(cat))+1])

        print('This comment is: ' + test_comment)                                
        print('This comment was placed in category ' + category + '. This means that we predict your comment will have between ' + leftBin + ' and ' + rightBin + ' recommendations.')
        return(category, leftBin, rightBin)

  
@app.route('/', methods=['GET'])    
def home():
    return render_template('home.html')    
    
@app.route('/comment/page/',methods=['GET', 'POST'])
def server():
    if request.method == 'GET':
        print("Got a request!")
        comment = str(request.args.get('comment', ''))
        category, leftBin, rightBin = predictor(comment)
        string = 'This comment was placed in category ' + category + '. This means that we predict your comment will have between ' + leftBin + ' and ' + rightBin + ' recommendations.'
        #payload = {comment: string}

        return render_template('comment.html', comment=comment, string=string)
    else:
        return '''<h1>Comment Quality Analyzer</h1>
<p>This is an error page. Did you type the link in correctly?</p>''' 

@app.route('/comment/api/',methods=['GET', 'POST'])
def webRequest():
    if request.method == 'GET':
        print("Got a request!")
        comment = str(request.args.get('comment', ''))
        category, leftBin, rightBin = predictor(comment)
        string = 'This comment was placed in category ' + category + '. This means that we predict your comment will have between ' + leftBin + ' and ' + rightBin + ' recommendations.'
        payload = {comment: string}

        return payload   

if __name__ == '__main__':
          app.run(host='0.0.0.0', port=8000, debug="true")
