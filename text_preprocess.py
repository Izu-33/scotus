import string
import re
from nltk.corpus import stopwords
import spacy

# nltk.download('stopwords')


months = ['Januanry', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']


def remove_stops(text, stops):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    words = text.split()
    
    # Remove stopwords
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = ' '.join(final)
    
    # Remove punctuation
    final = final.translate(str.maketrans('', '', string.punctuation))
    final = ''.join([i for i in final if not i.isdigit()])
    
    # Remove double whitespace
    while '  ' in final:
        final = final.replace('  ', ' ')
        
    return final

def clean_docs(docs, months):
    stops = stopwords.words('english')
    stops = stops + months
    clean_doc = remove_stops(docs, stops)
    return clean_doc

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out