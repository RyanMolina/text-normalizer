"""Module that generates the dataset."""
import os
import random
import time
from multiprocessing import Pool
import nltk.tokenize as tokenizer
from nltk.util import ngrams
from .textnoisifier import TextNoisifier


def csv_to_dict(file):
    """Convert csv to python dictionary."""
    d = {}
    with open(file, 'r') as f:
        rows = f.read().splitlines()
        for row in rows:
            k, v = row.split(',')
            d.setdefault(k, []).append(v)
    return d


def noisify(word):
    """Given a text, return noisy text.

    This function also wraps the instance method to be passed on Pool map.
    Since, multiprocessing.Pool.map() only accept single argument function.
    This function hides the 'self' in ntg.noisify().
    """
    return ntg.noisify(word)


def accent_style(word):
    return ntg.accent_style(word)


def trigram(text):
    tokens = text.split()
    if tokens >= 3:
        return list(ngrams(tokens, 3))


def collect_dataset(src, tgt, max_seq_len=50,
                    char_level_emb=False, augment_data=False,
                    shuffle=False, size=None):
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

    dataset_size = len(dataset)
    sent_number = 0
    start_time = time.time()
    print('   => Dataset Size: {}'.format(len(dataset)))
    print('   => collecting clean and noisy sentences')

    filename, _ = os.path.splitext(tgt)
    noisified_file = "{}.{}".format(filename, 'enc')

    with open(tgt, 'w', encoding="utf8") as decoder_file, \
            open(noisified_file, 'w', encoding='utf8') as encoder_file:

        for sentence in dataset:
            sent_number += 1

            if sent_number % 10000 == 0:
                speed = 10000 / (time.time() - start_time)
                print("      # {} "
                      "line/s: {:.2f} "
                      "ETA: {}".format(
                          sent_number,
                          speed,
                          (dataset_size - sent_number) / speed))
                start_time = time.time()
            # TODO (done): Allow punctuations on dataset (hyphen and apostrophe only).

            """ TODO: Create a model to detect named-entity
                   (but most likely named-entity occur less on dataset so the model will just copy it) """

            """ TODO: add of hyphen, when affix ends with consonant and the root word starts in vowel
                             (Fix later if the hyphens affect the overall accuracy) """
            # TODO: Concat seperator affix to its rootword

            # TODO: Word-level seq2seq to solve ng and nang (*Priority* this is a common mistake)

            # TODO (done): Generate edit distance for deletes using Peter Norvig's solution

            # TODO: Classifier for real words
            
            # TODO: Classifier for named-entity

            # Errors per word top 100 - Veritcal bar chart
            # Correct per word top 100 - Vertical bar chart
            # Perplexity - Line chart
            # Applied noisification in dataset - Radar chart and Pie chart
            # Accuracy/Error per model - Vertical bar chart
            # noisification category in informal words - pie chart

            clean_sentence = sentence[:max_seq_len]
            clean_sentence = clean_sentence[:clean_sentence.rfind(' ')]

            # Normalize the "r|d(oon, in, aw, ito mistakes" from Articles
            clean_sentence = ntg.raw_daw.sub(ntg.normalize_raw_daw,
                                             clean_sentence)

            # Normalize the contracted words (Siya ay -> Siya'y) from Articles
            clean_sentence = ntg.expand_pattern.sub(ntg.expand_repl,
                                                    clean_sentence)

            noisy_sentence = clean_sentence

            #  Contraction "Siya ay -> Siya'y"
            noisy_sentence = ntg.contract_pattern.sub(
                ntg.contract_repl, noisy_sentence)

            #  Noisify r|d(oon, in, aw, ito)
            noisy_sentence = ntg.raw_daw.sub(ntg.noisify_raw_daw,
                                             noisy_sentence)

            # Accent Style
            noisy_sentence = process_pool.map(accent_style,
                                              ntg.mwe_tokenizer(noisy_sentence.split()))

            noisy_sentence = noisy_sentence.split()

            try:
                sos = ntg.noisify(noisy_sentence[0], sos=True)
                noisy_sentence = process_pool.map(
                    noisify, noisy_sentence[1:])
                noisy_sentence.insert(0, sos)
                if random.getrandbits(1):
                    sos = ntg.noisify(noisy_sentence[0], sos=True)
                    noisy_sentence = process_pool.map(
                        noisify, noisy_sentence[1:])
                    noisy_sentence.insert(0, sos)
            except IndexError:
                # It is faster than checking length of the list
                pass

            noisy_sentence = ' '.join(noisy_sentence)
            if char_level_emb:
                clean_sentence = ' '.join(list(clean_sentence)) \
                                    .replace(' ' * 3, ' <space> ')
                noisy_sentence = ' '.join(list(noisy_sentence)) \
                                    .replace(' ' * 3, ' <space> ')

                clean_sentence = clean_sentence.replace('` `', '<lquotes>') \
                                       .replace("' '", '<rquotes>')
                noisy_sentence = noisy_sentence.replace('` `', '<lquotes>') \
                                        .replace("' '", '<rquotes>')

            if len(clean_sentence) > 1 and len(noisy_sentence) > 1:
                decoder_file.write(clean_sentence + "\n")
                encoder_file.write(noisy_sentence + "\n")

        decoder_file.truncate(decoder_file.tell() - 2)
        encoder_file.truncate(encoder_file.tell() - 2)

FILE_PATH = os.path.dirname(__file__)

accent_words_dict = csv_to_dict(
    os.path.join(FILE_PATH, 'accented_words.dic'))

phonetic_subwords_dict = csv_to_dict(
    os.path.join(FILE_PATH, 'phonetically_styled_subwords.dic'))

phonetic_words_dict = csv_to_dict(
    os.path.join(FILE_PATH, 'phonetically_styled_words.dic'))

with open(os.path.join(FILE_PATH, 'hyph_fil.tex'), 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_words_dict=accent_words_dict,
                    phonetic_words_dict=phonetic_words_dict,
                    phonetic_subwords_dict=phonetic_subwords_dict,
                    hyphenator_dict=hyphenator_dict)
