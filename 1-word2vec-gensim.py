import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line.split()
 
sentences = MySentences('./processed') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences=sentences, size=100, window=5, max_vocab_size=10000,  min_count=5, workers=4)
model.save('./results/gensim.model')

# m2 = gensim.models.Word2Vec.load('./results/gensim.model')
# print(m2.wv["قال"])