import random #for choosing the random response
import json
import pickle #for serialization
import numpy as np

import nltk #Natural Language Tool Kit
from nltk.stem import WordNetLemmatizer #Reduce the word to its stem

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json'). read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
#set - To eliminate duplicates
#sorted - Turn it back into a list, alphabetically
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))