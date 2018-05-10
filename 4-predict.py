from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import helper.paths as PATH
import helper.args as args
from helper.ticker import Ticker
import numpy as np

ticker = Ticker()
ticker.tick()

# generate a sequence from a language model
def generate_seq(model, word_index, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# map words to their integers
		encoded = np.array([word_index[w] for w in in_text.split()])
		# truncate to a fixed length (sliding window)
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

# load sequences into memory
with open(PATH.SEQUENCES, encoding='utf-8') as f:
	lines = f.read().split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model(PATH.MODEL)

# load the tokenizer
tokenizer = load(open(PATH.TOKENIZER, 'rb'))
word_index = tokenizer.word_index
del tokenizer

def fill(text):
	# replace 'NUM' with a random number
	for _ in range(text.split().count('<num>')):
		text = re.sub(r'<num>', str(randint(0, 2018)), text, flags=re.IGNORECASE, count=1)
	text = re.sub(r'<eos>', '.', text, flags=re.IGNORECASE)
	return text

def try_generate(seed_text='', generate_length=1, max_tries=10, one_sentence=False):
	
	count_removed = 1 # initially to enter the loop
	
	if one_sentence:
		generate_length = 1000 # acting as infinity
	
	generated = generate_seq(model, word_index, seq_length, seed_text, generate_length)
	while(count_removed > 0 and max_tries > 0):
		max_tries -= 1
		# remove consecutive duplicates
		split = generated.split(" ")
		count_removed = 0
		to_be_removed = list()
		for i in range(1, len(split)):
			if split[i-1] == split[i]:
				to_be_removed.append(i)
				count_removed += 1
		for i in to_be_removed:
			split[i] = ''
		
		generated = ' '.join([word for word in split if word != ''])
		print("Removed {} duplicates".format(count_removed))

		if one_sentence:
			s = generated.split()
			for j, word in enumerate(s):
				if word == '.':
					print("Stopped at dot")
					return ' '.join(s[:j+1])
		
		generated += ' ' + generate_seq(model, word_index, seq_length, seed_text + " " + generated, count_removed)
	
	return generated[:-1]



# Predict / Generate
generate_length = 20
f1 = open('./results/output-sentence.txt', mode="w", encoding='utf-8')
f2 = open('./results/output-{}.txt'.format(generate_length), mode="w", encoding='utf-8')

seed_text = ' '.join(lines[randint(0, len(lines))].split(' ')[:seq_length])

# Generate
generated1 = try_generate(seed_text=seed_text, one_sentence=True, max_tries=10)
generated2 = try_generate(seed_text=seed_text, generate_length=generate_length, max_tries=10)

output1 = fill(seed_text + '\n\n' + generated1)
output2 = fill(seed_text + '\n\n' + generated2)

f1.write(output1)
f2.write(output2)
	 
f1.close()
f2.close()
ticker.tock()