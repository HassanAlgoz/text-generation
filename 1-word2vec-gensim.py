import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec, Phrases
import os
import helper.paths as PATH
import helper.args as args

class SentenceGenerator(object):
    def __init__(self, filepath):
        self.filepath = filepath
 
    def __iter__(self):
        # for fname in os.listdir(self.dirname):
        for line in open(self.filepath, encoding='utf-8'):
            yield line.split()
 
sentences = SentenceGenerator(PATH.CLEAN_TEXT) # a memory-friendly iterator

# Note that there is a gensim.models.phrases module which lets you automatically detect 
# phrases longer than one word. Using phrases, you can learn a word2vec model where “words”
#  are actually multiword expressions, such as new_york_times or financial_crisis:
bigram_transformer = Phrases(sentences)
model = Word2Vec(bigram_transformer[sentences], size=args.EMBEDDING_HIDDEN_SIZE, window=args.EMBEDDING_WINDOW_SIZE, max_vocab_size=args.MAX_VOCAB_SIZE,  min_count=5, workers=4)

# Save the model
model.save(PATH.GENSIM_MODEL)

# model.wv.save_word2vec_format('./results/gensim.wv')
# m2 = gensim.models.Word2Vec.load('./results/gensim.model')
# print(m2.wv["قال"])