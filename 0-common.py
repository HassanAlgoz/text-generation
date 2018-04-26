from helper import arabic_const
import re
import collections
import sklearn
import time
import pickle
import helper.paths as PATH
import helper.args as args

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

# Count the most common n tokens
counter = collections.Counter()
filtered_text = ''
with open(PATH.TEXT, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        tokens = filter_text(line).replace('\n', ' ').split()
        tokens = [t for t in tokens if t.strip() != '']
        filtered_text += ' '.join(tokens)
        counter.update(tokens)
        line = f.readline()

# Get only the most common n
most_common = list()
for word, count in counter.most_common(args.MAX_VOCAB_SIZE):
    most_common.append(word)
print("most common:", len(most_common))
del counter

with open(PATH.MOST_COMMON, mode="w", encoding='utf-8') as f:
    f.write('\n'.join(most_common))

clean_text = ' '.join([t for t in filtered_text.split() if t in most_common])

# Write them to clean_text.txt
with open(PATH.CLEAN_TEXT, mode='w', encoding='utf-8') as f:   
    f.write(clean_text)

# Display elapsed time
endTime = time.time()
print("Elapsed Time: {}s".format(endTime - startTime))