import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding
from keras.callbacks import LambdaCallback

# load sequences
with open('./processed/sequences.txt', encoding='utf-8') as f:
	texts = f.read().split('\n')

# Load tokenizer and convert word sequences to their integers
tokenizer = pickle.load(open('./results/tokenizer.pkl', 'rb'))
sequences = np.asarray(tokenizer.texts_to_sequences(texts))
vocab_size = len(tokenizer.word_index) + 1
del tokenizer
print(sequences[:5])


# Embeddings
embeddings = pickle.load(open('./results/embeddings.pkl', 'rb'))
embedding_dim = embeddings.shape[1]
print('embeddings shape: {}'.format(embeddings.shape))

# separate into input and output
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# Define model
model = Sequential([
    # Input to Embedding is integers, used to select the embedding vector corresponding to the word at that index.
    Embedding(vocab_size, embedding_dim, weights=[embeddings], input_length=seq_length, trainable=False),
    GRU(100, return_sequences=True),
    GRU(100),
    Dense(100, activation='relu'),
    Dense(vocab_size, activation='softmax'),
])
model.summary()
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# callback
def on_epoch_end(epoch, logs):
    # save the model to a file
    model.save('./results/model.h5')

callbacks = LambdaCallback(on_epoch_end=on_epoch_end)

# fit model
model.fit(X, y, batch_size=10, epochs=100, callbacks=[callbacks])

