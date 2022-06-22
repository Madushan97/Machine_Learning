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

# save in the file (Write Binary)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

# end of the machine learning part

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = documents[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag, output_row)

random.shuffle(training)
training = np.array(training)

training_x = list(training[:, 0])
training_y = list(training[:, 1])