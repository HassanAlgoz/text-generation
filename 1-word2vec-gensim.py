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
ticker.tick("Building Word2Vec")
bigram_transformer = Phrases(sentences)
model = Word2Vec(bigram_transformer[sentences], size=args.EMBEDDING_HIDDEN_SIZE, window=args.EMBEDDING_WINDOW_SIZE, max_vocab_size=args.MAX_VOCAB_SIZE,  min_count=5, workers=os.cpu_count())
ticker.tock()

ticker.tick("Gensim Training")
model.train(bigram_transformer[sentences], total_words=num_words, epochs=args.EMBEDDING_EPOCHS)
ticker.tock()

# Save the model
model.save(PATH.GENSIM_MODEL)


# model.wv.save_word2vec_format('./results/gensim.wv')
# m2 = gensim.models.Word2Vec.load('./results/gensim.model')
# print(m2.wv["قال"])

ticker.tock()