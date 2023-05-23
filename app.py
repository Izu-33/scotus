import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import (
    LSTM,
    Dropout,
    Bidirectional,
    Dense,
    Embedding
)
from tensorflow.keras.models import Sequential
import spacy
import en_core_web_sm
from text_preprocess import remove_stops, clean_docs, lemmatization
import pickle


# Load the trained model
model = pickle.load(open('models/model.pickle', 'rb'))
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

st.write("""
# SCOTUS Prediction App

This app predicts the **verdict of a Supreme Court case**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Case Facts')

# User input for case description
facts = st.text_area("")


issue_area = [
            "Criminal Procedure",
            "Civil Rights",
            "Economic Activity",
            "First Amendment",
            "Judicial Power",
            "Due Process",
            "Federalism",
            "Privacy",
            "Unions",
            "Federal Taxation",
            "Attorneys",
            "Miscellaneous",
            "Private Action",
            "Interstate Relations"
            ]

issue = st.radio('Issue Area', issue_area)

if st.button("Predict"):
    # Preprocess the case description (similar to training data preprocessing)
    cleaned_facts = lemmatization(clean_docs(facts))

    # Apply feature extraction (e.g., TF-IDF) on the preprocessed case
    vectorized_facts = vectorizer.transform([cleaned_facts])



    # Make prediction using the loaded model
    # prediction = model.predict(transformed_case)[0]

# Main Panel
# Print specified input parameters
st.header('Specified Input parameters')
st.write('---')