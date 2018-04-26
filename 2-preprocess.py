import re
import string
from keras.preprocessing.text import Tokenizer
import pickle
 
# load text
with open('./processed/clean_text.txt', encoding='utf-8') as f:
	text = f.read()
print(text[:200])

tokens = text.split()
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# Tokenizer
tokenizer = Tokenizer(filters="", oov_token="UNK") # we don't filter any characters, we need commans full stops and everything.
tokenizer.fit_on_texts(tokens)
# save the tokenizer
pickle.dump(tokenizer, open('./processed/tokenizer.pkl', 'wb'))
del tokenizer
 
# organize into sequences of tokens
length = 15 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

with open('./processed/sequences.txt', mode='w', encoding='utf-8') as f:
	f.write('\n'.join(sequences))
