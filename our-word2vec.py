import numpy as np
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding
from keras.callbacks import LambdaCallback
import string
import re

# load doc into memory
words = ''
with open('./processed/clean_text.txt', encoding='utf-8') as f:
	words = f.read().split()

# One-hot encode all words
tokenizer = Tokenizer(filters="", oov_token="UNK") # we don't filter any characters, we need commans full stops and everything.
tokenizer.fit_on_texts(words)
vocab_size = len(tokenizer.word_index) + 1
one_hot_matrix = tokenizer.texts_to_matrix(words)
# num_examples = len(one_hot_matrix)

# save the tokenizer
dump(tokenizer, open('./processed/tokenizer.pkl', 'wb'))
del tokenizer

# Skip-Gram
# prepare examples
X = []
y = []
window_size = 5
for index, row in enumerate(one_hot_matrix):
    for j in range(max(index - window_size, 0), min(index + window_size, len(one_hot_matrix))):
        if index != j:
            X.append(row)
            y.append(one_hot_matrix[j])

# release memory
del one_hot_matrix

# convert to numpy arrays
X = np.asarray(X)
y = np.asarray(y)

# Define the NN
model = Sequential([
    Dense(50, input_shape=(vocab_size,)),
    Dense(vocab_size, activation='softmax')
])
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print model's summary
model.summary()

# callback
def on_epoch_end(epoch, logs):
    # Save embeddings (weights of the first hidden layer)
	dump(model.get_weights()[0], open('./results/embeddings.pkl', 'wb'))

callbacks = LambdaCallback(on_epoch_end=on_epoch_end)

# fit model
model.fit(X, y, batch_size=10, epochs=100, callbacks=[callbacks])