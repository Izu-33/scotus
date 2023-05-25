import streamlit as st
import pandas as pd
# import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
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
from text_preprocess import remove_stops, clean_docs, lemmatization, months
import joblib
import tensorflow as tf


# Load the trained model
vectorizer = joblib.load(open('vectorizers/vectorizer.joblib', 'rb'))
model = tf.keras.models.load_model('models/model.h5')

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

st.title("""⚖️ SCOTUS Prediction App ⚖️
#### This app predicts the **verdict of a Supreme Court case**
""")
st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
   st.subheader('Specify Case Facts')

   # User input for case description
   facts = st.text_area("", height=340)

with col2:
    st.subheader('Specify Issue Area')
    issue = st.radio('', issue_area)

    if issue == "Criminal Procedure":
        crim_proc = 1
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Civil Rights":
        crim_proc = 0
        civil_rights = 1
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Economic Activity":
        crim_proc = 0
        civil_rights = 0
        eco_act = 1
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0  
    elif issue == "First Amendment":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 1
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Judicial Power":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 1
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Due Process":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 1
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Federalism":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 1
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Privacy":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 1
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0  
    elif issue == "Unions":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 1
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Federal Taxation":
        crim_proc = 1
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 1
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Attorneys":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 1
        misc = 0
        priv_act = 0
        inter_rel = 0
    elif issue == "Miscellaneous":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 1
        priv_act = 0
        inter_rel = 0  
    elif issue == "Private Action":
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 1
        inter_rel = 0
    elif issue == "Interstate Relations":
        crim_proc = 1
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 1
    else :
        crim_proc = 0
        civil_rights = 0
        eco_act = 0
        first_amend = 0
        judicial = 0
        due_process = 0
        federalism = 0
        privacy = 0
        unions = 0
        fed_tax = 0
        attorneys = 0
        misc = 0
        priv_act = 0
        inter_rel = 0

var_list = [
    crim_proc,
    civil_rights,
    eco_act,
    first_amend,
    judicial,
    due_process,
    federalism,
    privacy,
    unions,
    fed_tax,
    attorneys,
    misc,
    priv_act, 
    inter_rel
        ]

if st.button("Predict"):
    # Preprocess the case description (similar to training data preprocessing)
    cleaned_facts = lemmatization(clean_docs(facts, months))

    # Apply feature extraction (e.g., TF-IDF) on the preprocessed case
    vectorized_facts = vectorizer.transform(cleaned_facts)

    arr = vectorized_facts.toarray()

    dataframe = pd.DataFrame(data=arr, columns=vectorizer.get_feature_names_out())
    
    for k, v in zip(issue_area, var_list):
        dataframe[k] = v

    pca = PCA(n_components=64)
    pca_fit = pca.fit_transform(dataframe)
    pca_df = pd.DataFrame(data = pca_fit)
    prediction = model.predict(pca_df)[0]

    st.write("Prediction:")
    if prediction == 1:
        st.write("👩🏽‍⚖️... First party has a high chance of winning.")
    else:
        st.write("👩🏽‍⚖️... Second party has a high chance of winning.")