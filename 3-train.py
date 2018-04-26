import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras import callbacks
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import os
import helper.paths as PATH

# load sequences
with open(PATH.SEQUENCES, encoding='utf-8') as f:
	texts = f.read().split('\n')

# Load tokenizer and convert word sequences to their integers
tokenizer = pickle.load(open(PATH.TOKENIZER, 'rb'))
sequences = np.array(tokenizer.texts_to_sequences(texts))
vocab_size = len(tokenizer.word_index) + 1
print(sequences[:5])
word_index = tokenizer.word_index
# del tokenizer

# Embeddings
word2vec = gensim.models.Word2Vec.load(PATH.GENSIM_MODEL)
word_vectors = word2vec.wv
vector_size = word_vectors.vector_size
del word2vec

# https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
embedding_matrix = np.zeros((vocab_size, word_vectors.vector_size))
for word, i in word_index.items():
    if word in word_vectors.vocab:
        embedding_matrix[i] = np.transpose(word_vectors.word_vec(word))
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
del word_vectors

# separate into input and output
print('separate into input and output...')
X, y = sequences[:, :-1], sequences[:, -1]
num_sequences = len(sequences)
del sequences
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# Define model
if os.path.exists(PATH.MODEL):
    print("loading a saved model...")
    model = load_model(PATH.MODEL)
else:
    print("starting from a new model...")
    model = Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=vector_size,
            input_length=seq_length,
            trainable=False,
            weights=[embedding_matrix]
        ),
        layers.GRU(vector_size),
        layers.Dense(vocab_size, activation='softmax'),
    ])
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

model.summary()


# save model after each epoch
checkpoint_cb = callbacks.ModelCheckpoint(PATH.MODEL)
# if the value monitored doesn't improve in 10 epochs, stop training.
earlystop_cb = callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')
# fit model
model.fit(X, y, batch_size=10, epochs=1000, verbose=2, callbacks=[checkpoint_cb, earlystop_cb])

