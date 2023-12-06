import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer



def text_cleaning(text):
    ## converting to lowercase

    text = text.lower()

    #  Tokenization

    text_ = nltk.word_tokenize(text)

    # Removing special character

    y = []
    for i in text_:
        if i.isalnum():
            y.append(i)

    # Removing stopwords and punctuation
    x = []
    stop_words = set(stopwords.words('english'))
    for i in y:
        if i not in stop_words and i not in string.punctuation:
            x.append(i)

    # Performing stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in x]

    return ' '.join(words)

tfidf = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

st.title('Email/SMS Classifier')
input_sms = st.text_input("Enter the message : ")

if st.button("Predict"):

    #1. Preprocess the data

    transformed_text = text_cleaning(input_sms)

    #2. Vectorize the data

    text = tfidf.transform([transformed_text])

    #3. Predict

    output = model.predict(text)[0]

    #4. Display

    if output == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")