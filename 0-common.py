from helper import arabic_const
import re
import collections
import sklearn
import time
import pickle
import helper.paths as PATH
import helper.args as args
import numpy as np

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
    # separate punctuation from words
    split = text.split(' ')
    for i, word in enumerate(split):
        for j, char in enumerate(word):
            if is_punctuation(char):
                split[i] = word[:j] + ' ' + word[j:]
    text = ' '.join(split)
    return text

startTime = time.time()

# Count the most common n tokens
counter = collections.Counter()
all_tokens = list()
with open(PATH.TEXT, mode='r', encoding='utf-8') as f:
    for line in f:
        tokens = filter_text(line).replace('\n', ' ').split()
        tokens = [t for t in tokens if len(t) > 0]
        all_tokens.extend(tokens)
        print(tokens)

    counter.update(all_tokens)

# Get only the most common n
most_common = list()
for word, count in counter.most_common(args.MAX_VOCAB_SIZE):
    most_common.append(word)
print("most common:", len(most_common))
del counter

with open(PATH.MOST_COMMON, mode="w", encoding='utf-8') as f:
    f.write('\n'.join(most_common))

clean_text = ' '.join([t for t in all_tokens if t in most_common])

# Write them to clean_text.txt
with open(PATH.CLEAN_TEXT, mode='w', encoding='utf-8') as f:   
    f.write(clean_text)

# Display elapsed time
endTime = time.time()
print("Elapsed Time: {}s".format(endTime - startTime))