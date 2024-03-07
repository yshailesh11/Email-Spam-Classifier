import streamlit as st
import pickle
import string
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def trans_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for each in text:
        if(each.isalnum()):
            y.append(each)

    text = y[:]
    y.clear()

    for each in text:
        if each not in stopwords.words('english') and each not in string.punctuation:
            y.append(each)

    text = y[:]
    y.clear()

    for each in text:
        y.append(ps.stem(each))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email Spam Classifier')

input_text = st.text_area('Enter the message')

if st.button('Show Result'):

    tf_msg = trans_text(input_text)
    vec_msg = tfidf.transform([tf_msg])
    result  = model.predict(vec_msg)[0]

    if(result==1):
        st.header("Spam")
    else:
        st.header("Not Spam")


