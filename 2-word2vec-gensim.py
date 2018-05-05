import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec, Phrases
import os
import helper.paths as PATH
import helper.args as args
from helper.ticker import Ticker

ticker = Ticker()
ticker.tick()

class SentenceGenerator(object):
    def __init__(self, filepath):
        self.filepath = filepath
 
    def __iter__(self):
        # for fname in os.listdir(self.dirname):
        for line in open(self.filepath, encoding='utf-8'):
            yield line.split()
 
sentences = SentenceGenerator(PATH.CLEAN_TEXT) # a memory-friendly iterator

num_words = 0
with open(PATH.CLEAN_TEXT, encoding='utf-8') as f:
    for line in f:
        num_words += len(line.split())

# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.WuT-OYhuaiM
# Note that there is a gensim.models.phrases module which lets you automatically detect 
# phrases longer than one word. Using phrases, you can learn a word2vec model where “words”
#  are actually multiword expressions, such as new_york_times or financial_crisis:

ticker.tick("bigram phrases model")
bigram = Phrases(sentences)
ticker.tock()

if os.path.exists(PATH.GENSIM_MODEL):
    print("load an existing gensim model...")
    model = Word2Vec.load(PATH.GENSIM_MODEL)
else:
    ticker.tick("Building Word2Vec")
    model = Word2Vec(bigram[sentences], size=args.EMBEDDING_HIDDEN_SIZE, window=args.EMBEDDING_WINDOW_SIZE, max_vocab_size=args.MAX_VOCAB_SIZE,  min_count=5, workers=os.cpu_count())
    ticker.tock()

ticker.tick("Gensim Training ({} epochs)".format(args.EMBEDDING_EPOCHS))
model.train(bigram[sentences], total_words=num_words, epochs=args.EMBEDDING_EPOCHS)
ticker.tock()

# Save the model
model.save(PATH.GENSIM_MODEL)

ticker.tock()