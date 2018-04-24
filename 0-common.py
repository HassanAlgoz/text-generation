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
fin  = open('./raw-data/text.txt', mode='r', encoding='utf-8')
fout = open('./processed/clean_text.txt', mode='w', encoding='utf-8')
line = fin.readline()
while line:
    clean_line = filter_text(line)
    fout.write(clean_line)
    tokens = clean_line.replace('\n', ' ').split()
    tokens = [t for t in tokens if t.strip() != '']
    counter.update(tokens)
    line = fin.readline()
fin.close()
fout.close()

# pickle.dump(counter, open('words_counter.pkl', 'wb'))

# most_common = counter.most_common(10000)
# f1 = open('10000_most_common_words.txt', 'w', encoding='utf-8')
# f2 = open('10000_most_common_words_WORDS_ONLY.txt', 'w', encoding='utf-8')
# for word, count in most_common:
#     f1.write('{}\t{}\n'.format(word, count))
#     f2.write('{}\n'.format(word))
# f1.close()
# f2.close()

# Display elapsed time
endTime = time.time()
print("Elapsed Time: {}s".format(endTime - startTime))