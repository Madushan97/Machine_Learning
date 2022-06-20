import random
import json
import pickle

import nltk
nltk.download('punkt')
import numpy as np

from nltk.stem import WordNetLemmatizer       # wor, works, working, worked are same

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

# reading the json content
intents = json.loads(open('intents.json').read())

# creating empty list
words = []
classes = []
documents = []
ignore_latters = ['?', '!', '.', ',']

# going through the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)              # I am john -> split the sentence into tokens
        words.append(word_list)                         # word_list into words list
        documents.append((word_list, intent['tag']))
        # checking class is whether in the class list or not
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)
