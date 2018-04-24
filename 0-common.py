from helper import arabic_const
import re
import collections
import sklearn
import time
import pickle

def filter_text(text):
    # strip tashkeel
    text = arabic_const.HARAKAT_PAT.sub('', text)
    # strip tatweel
    text = re.sub(u'[%s]' % arabic_const.TATWEEL, '', text)
    # replace english
    text = re.sub(r'[a-zA-Z]+', 'ENG', text)
    # replace numbers
    text = re.sub(r'\d+', 'NUM', text)
    # remove [[NUM]]
    text = re.sub(r'\[\[NUM\]\]', '', text)
    return text

startTime = time.time()

# Count the most common 10,000 tokens
counter = collections.Counter()
clean_text = ''
with open('./raw-data/text.txt', mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        tokens = filter_text(line).replace('\n', ' ').split()
        tokens = [t for t in tokens if t.strip() != '']
        clean_text += ' '.join(tokens)
        counter.update(tokens)
        line = f.readline()

# Get only the most common 10,000
most_common = list()
for word, count in counter.most_common(10000):
    most_common.append(word)

del counter
clean_tokens = [t for t in clean_text.split() if t in most_common]

# Write them to clean_text.txt
with open('./processed/clean_text.txt', mode='w', encoding='utf-8') as f:   
    for token in clean_tokens:
        f.write(token + ' ')

# Display elapsed time
endTime = time.time()
print("Elapsed Time: {}s".format(endTime - startTime))