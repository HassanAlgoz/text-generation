import re
import collections
import sklearn
import time
import pickle
import numpy as np

from helper import arabic_const
import helper.paths as PATH
import helper.args as args
from helper.ticker import Ticker

ticker = Ticker()
ticker.tick()

def is_punctuation(char):
    code = ord(char)
    if (ord('\u0020') <= code and code <= ord('\u0040')) or code == ord('\u061F') or code == ord('\u060C') or code == ord('\u061B') or code == ord('\u061F'):
        return True

def filter_text(text):
    # strip tashkeel
    text = arabic_const.HARAKAT_PAT.sub('', text)
    # strip tatweel
    text = re.sub(u'[%s]' % arabic_const.TATWEEL, '', text)
    # replace english
    text = re.sub(r'[a-zA-Z]+', 'ENG', text)
    # replace numbers
    text = re.sub(r'\d+', 'NUM', text)
    # separate punctuation from words (trailing punctuation only)
    split = text.split(' ')
    for i, word in enumerate(split):
        for j, char in enumerate(word):
            if is_punctuation(char):
                split[i] = word[:j] + ' ' + word[j:]
    text = ' '.join(split)
    return text


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
        tokens = filter_text(line + ' .').split()
        tokens = [token for token in tokens if len(token) > 0]
        all_tokens.extend(tokens)

    # Separate "waw و", notice: passing a copy of all_tokens not a reference
    new_tokens = separate_waw(list(all_tokens))
    # new_tokens = all_tokens

    counter.update(new_tokens)

# Get only the most common n
most_common = list()
for word, count in counter.most_common(args.MAX_VOCAB_SIZE):
    most_common.append(word)
print("most common:", len(most_common))
del counter

with open(PATH.MOST_COMMON, mode="w", encoding='utf-8') as f:
    f.write('\n'.join(most_common))

# When writing clean_text, separate the و if the token without و is in most_common
# Write clean_text to file
with open(PATH.CLEAN_TEXT, mode='w', encoding='utf-8') as f:   
    for token in all_tokens:
        if token in most_common:
            f.write(token)
        elif token[0] == 'و' and token[1:] in most_common:
            f.write(token[0] + ' ' + token[1:])
    
        if token == ".":
            f.write('\n')
        else:
            f.write(' ')

ticker.tock()