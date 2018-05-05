# RNN Sequence Length
SEQUENCE_LENGTH = 20

# N most common words
MAX_VOCAB_SIZE = 10000

# word2vec (gensim)
EMBEDDING_HIDDEN_SIZE = 150
EMBEDDING_WINDOW_SIZE = 6
EMBEDDING_EPOCHS = 1000

# Training
BATCH_SIZE = 10
EPOCHS = 1000
OPTIMIZER = 'adagrad'
# Callbacks
VERBOSE = 2
PATIENCE = 15
PERIOD = 15

# Prediction: number of words to generate
GENERATE_LENGTH = 50