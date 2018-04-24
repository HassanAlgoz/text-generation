from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

# load doc into memory
with open('sequences.txt', encoding='utf-8') as f:
	lines = f.read().split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

def fill(text):
	# replace 'NUM' with a random number
	text = re.sub(r'NUM', str(randint(0, 2018)), text, flags=re.IGNORECASE)
	return text

with open('./output.txt', mode="w", encoding='utf-8') as f:
	# for _ in range(10):
	seed_text = ' '.join(lines[randint(0, len(lines))].split(' ')[:seq_length])
	output = seed_text + ' ' + generate_seq(model, tokenizer, seq_length, seed_text, 50)
	output = fill(output)
	f.write(output + '\n')
	# print(output + '\n')