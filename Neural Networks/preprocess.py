from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import numpy as np 
import pandas as pd 
import re
import nltk

def neural(text):
    text = str(text)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text

def preprocess_neural(q1,q2,dup):
    q1_preprocessed, q2_preprocessed = [], []
    for i in range(len(q1)):
        q1_preprocessed.append(neural(q1[i]))
        q2_preprocessed.append(neural(q2[i]))
    # store in file called preprocessed_neural.csv
    df = pd.DataFrame({'question1': q1_preprocessed, 'question2': q2_preprocessed, 'is_duplicate': dup})
    df.to_csv('preprocessed_neural.csv', index=False)
    return q1_preprocessed, q2_preprocessed