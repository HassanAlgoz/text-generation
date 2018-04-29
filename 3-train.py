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
import sys
import helper.paths as PATH
import helper.args as args
from helper.tick import Tick

tick = Tick()

# load sequences
with open(PATH.SEQUENCES, encoding='utf-8') as f:
	texts = f.read().split('\n')

num_sequences = len(texts)
# Load tokenizer and convert word sequences to their integers
tokenizer = pickle.load(open(PATH.TOKENIZER, 'rb'))
sequences = np.array(tokenizer.texts_to_sequences(texts))
vocab_size = len(tokenizer.word_index) + 1
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
            input_length=args.SEQUENCE_LENGTH,
            trainable=False,
            weights=[embedding_matrix]
        ),
        layers.GRU(50),
        layers.Dense(vocab_size, activation='softmax'),
    ])
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=args.OPTIMIZER, metrics=['accuracy'])

model.summary()

# save model after each epoch
checkpoint_cb = callbacks.ModelCheckpoint(PATH.MODEL)
# if the value monitored doesn't improve in 10 epochs, stop training.
earlystop_cb = callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')

# separate into input and output
# print('separate into input and output...')
# X, y = sequences[:, :-1], sequences[:, -1]
# del sequences
# y = to_categorical(y, num_classes=vocab_size)

# fit model
# print("fitting...")
# model.fit(X, y, batch_size=args.BATCH_SIZE, epochs=args.EPOCHS, verbose=2, callbacks=[checkpoint_cb, earlystop_cb])
# steps_per_epoch=num_sequences // args.BATCH_SIZE

def sequence_generator():
    while True:
        with open(PATH.SEQUENCES, encoding='utf-8') as f:
            X = np.zeros((args.BATCH_SIZE, args.SEQUENCE_LENGTH))
            y = np.zeros((args.BATCH_SIZE))
            i = 0
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                encoded = np.array(tokenizer.texts_to_sequences([line])[0])
                np.put(X, i, encoded[:-1])
                np.put(y, i, encoded[-1])

                i = (i+1) % args.BATCH_SIZE
                if i == 0:
                    yield X, to_categorical(y, num_classes=vocab_size)

num_workers = os.cpu_count()
use_multiprocessing = sys.platform.startswith('linux') # use_multiprocessing doesn't work on windows.
print("fitting (generator)...")
# https://keras.io/models/sequential/#sequential-model-methods
model.fit_generator(
    generator=sequence_generator(),
    # validation_data=
    epochs=args.EPOCHS,
    steps_per_epoch=num_sequences // args.BATCH_SIZE,
    verbose=2,
    callbacks=[checkpoint_cb, earlystop_cb],
    workers=num_workers,
    use_multiprocessing=use_multiprocessing
)

tick.tock()