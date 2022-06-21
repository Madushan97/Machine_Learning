import random
import json
import pickle

import nltk

import numpy as np

from nltk.stem import WordNetLemmatizer       # wor, works, working, worked are same

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

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
        words.extend(word_list)                              # word_list into words list
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:                     # checking class is whether in the class list or not
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_latters]
words = sorted(set(words))                                  # eliminate the duplicate

# avoiding duplicates in classes

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

print(words)