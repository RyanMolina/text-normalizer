import re
from multiprocessing import Pool
import random
import time
from nltk.util import ngrams


spaces = re.compile(r'\s+')
ascii_only = re.compile(r'[^\x00-\x7f]')

# MANILA, Philippines - <Article>
news_location = re.compile(r'^(\w+\s\w+|\w+),\s\w+\s-\s')


def trigram(text):
    tokens = text.split()
    if len(tokens) >= 3:
        return list(ngrams(tokens, 3))


def unigram(text):
    tokens = text.split()
    if len(tokens) >= 1:
        return list(ngrams(tokens, 1))

def bigram(text):
    tokens = text.split()
    if len(tokens) >= 2:
        return list(ngrams(tokens, 2))


def quad_2_ngram(text):
    tokens = text.split()
    ngram = [grams
             for i in range(4, len(tokens))
             for grams in ngrams(tokens, i)]
    return ngram


def manipulate(src, shuffle=False, augment_data=False, size=None):
    """Generate a parallel clean and noisy text from a given clean text."""
    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        dataset = [content.strip()
                     for content in contents.splitlines()
                     if content]
    print('# Initialize multiprocess.Pool()')
    if shuffle:
        print('# 1st shuffle of dataset')
        random.shuffle(dataset)

    if augment_data:
        print('  [+] Augment dataset using ngram of dataset.')
        dataset.extend(random.sample(augment_dataset(dataset, end=50000), len(dataset)))

    if shuffle:
        print('# 2nd shuffle of dataset')
        random.shuffle(dataset)

    if size:
        print('# Truncate dataset to desired dataset size.')
        dataset = dataset[:size]

    print('   => Dataset Size: {}'.format(len(dataset)))
    print('   => collecting clean and noisy sentences')

    return dataset

def augment_dataset(dataset, start=0, end=50000):
        process_pool = Pool()
        n_grams = process_pool.map(quad_2_ngram, dataset[start:end])
        print('    [-] Flattening n-grams...')
        n_grams = [' '.join(list(gram))
                            for grams in n_grams
                            if grams
                            for gram in grams
                            if gram]
        print('    [-] Add ngrams to dataset')
        process_pool.close()
        return n_grams