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


def manipulate(src, shuffle=False, augment_data=False, size=None):
    """Generate a parallel clean and noisy text from a given clean text."""
    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        dataset = [content.strip()
                     for content in contents.splitlines()
                     if content]
    print('# Initialize multiprocess.Pool()')
    process_pool = Pool()

    if shuffle:
        print('# 1st shuffle of dataset')
        random.shuffle(dataset)

    if augment_data:
        print('  [+] Augment dataset using trigram of dataset.')
        print('      Get the trigrams of 10% of the dataset.')
        trigrams = process_pool.map(trigram, dataset[:int(len(dataset)*0.10)])
        print('    [-] Flattening trigrams...')
        dataset_trigrams = [' '.join(list(gram))
                            for grams in trigrams
                            if grams
                            for gram in grams
                            if gram]
        print('    [-] Shuffling trigrams')
        random.shuffle(dataset_trigrams)
        dataset.extend(dataset_trigrams)
        print('    [-] Add ngrams to dataset')
        dataset_size = len(dataset)
        print('  [+] New dataset size: {}'.format(dataset_size))

    if shuffle:
        print('# 2nd shuffle of dataset')
        random.shuffle(dataset)

    if size:
        print('# Truncate dataset to desired dataset size.')
        dataset = dataset[:size]

    print('   => Dataset Size: {}'.format(len(dataset)))
    print('   => collecting clean and noisy sentences')

    return dataset

