from email import message
import pickle
from flask import Flask,render_template,request,url_for
import string
import sklearn
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import os
print(os.listdir())
tfidf =pickle.load(open("vectorizer.pkl","rb"))
model =pickle.load(open("bnbmodel.pkl","rb"))
app = Flask(__name__)

def transForm_text(text):
    text=text.lower()
    text=word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text =y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text =y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        input_sms=request.form['message']
        transformed_sms =transForm_text(input_sms)
        vector_input =tfidf.transform([transformed_sms])
        my_prediction =model.predict(vector_input)[0]
        print(my_prediction)
        pred={0:"not spam ",1:"spam"}
        message=""
        if my_prediction==1:
            message=pred[1]
        else:
            message=pred[0]        
        return render_template('index.html',prediction=message)
if __name__=="__main__":
    app.run(debug=True)