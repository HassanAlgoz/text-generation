import re
import collections
import sklearn
import time
import pickle
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer

from helper import arabic_const
import helper.paths as PATH
import helper.args as args
from helper.ticker import Ticker

ticker = Ticker()
ticker.tick()

def filter_text(text):
    # replace english
    # text = re.sub(r'[a-zA-Z]+', 'ENG', text)
    # strip tashkeel
    text = arabic_const.HARAKAT_PAT.sub('', text)
    # strip tatweel
    text = re.sub(u'[%s]' % arabic_const.TATWEEL, '', text)
    # split punctuation
    split = re.split(r'(\W+)', text)
    text = ' '.join([word.strip() for word in split if len(word.strip()) > 0])

    # replace numbers
    text = re.sub(r'\d+', '<num>', text)
    # replace '.' at the end of the line with '<eos>' (end of sentece)
    text = re.sub(r' \.$', '<eos>', text)

    return text.lower()

def separate_waw(all_tokens):
    # Step 1: find such word (token) and insert a space in-between
    # ["والغراب"] -> ["و الغراب"]
    for i, token in enumerate(all_tokens):
        if token[0] == 'و' and token in set(all_tokens):
            all_tokens[i] = token[0] + ' ' + token[1:]
    
    # Step 2: separate into two elements
    # ["و الغراب"] -> ["و", "الغراب"]
    new_tokens = list()
    for token in all_tokens:
        if len(token) > 1 and token[1] == ' ':
            new_tokens.append(token[0])
            new_tokens.append(token[2:])
        else:
            new_tokens.append(token)
    
    return new_tokens

# Count the most common n tokens
counter = collections.Counter()
all_tokens = list()
with open(PATH.TEXT, mode='r', encoding='utf-8') as f:
    for line in f:
        if line[0] == '\n':
            continue
        tokens = filter_text(line).split(" ")
        tokens = [token for token in tokens if len(token) > 0]
        all_tokens.extend(tokens)

    # Separate "waw و", notice: passing a copy of all_tokens not a reference
    # new_tokens = separate_waw(list(all_tokens))

    counter.update(all_tokens)

# Get only the most common n
most_common = list()
for word, count in counter.most_common(args.MAX_VOCAB_SIZE):
    most_common.append(word)
print("most common:", len(most_common))
del counter

with open(PATH.MOST_COMMON, mode="w", encoding='utf-8') as f:
    f.write('\n'.join(most_common))

# Write clean_text file
clean_tokens = [token for token in all_tokens if token in most_common]
with open(PATH.CLEAN_TEXT, mode='w', encoding='utf-8') as f:
    for token in clean_tokens:
        f.write(token)
        
        if token == ".":
            f.write('\n')
        else:
            f.write(' ')

print(clean_tokens[:20])
print('Total Tokens: %d' % len(clean_tokens))
print('Unique Tokens: %d' % len(set(clean_tokens)))

# Tokenizer
tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(clean_tokens)
# save the tokenizer
pickle.dump(tokenizer, open(PATH.TOKENIZER, 'wb'))
del tokenizer
 
# organize into sequences
length = args.SEQUENCE_LENGTH + 1
sequences = list()
for i in range(length, len(clean_tokens)):
	sequences.append(' '.join(clean_tokens[i-length:i]))
    
print('Total Sequences: %d' % len(sequences))

with open(PATH.SEQUENCES, mode='w', encoding='utf-8') as f:
	f.write('\n'.join(sequences))

ticker.tock()