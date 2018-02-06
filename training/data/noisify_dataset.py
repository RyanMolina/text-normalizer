"""Module that generates the dataset."""
import os
import random
import time
import re
from multiprocessing import Pool
# from utils import tokenizer
import nltk.tokenize as tokenizer
from nltk.tokenize.moses import MosesDetokenizer
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


def noisify(text):
    """Given a text, return noisy text.

    This function also wraps the instance method to be passed on Pool map.
    Since, multiprocessing.Pool.map() only accept single argument function.
    This function hides the 'self' in ntg.noisify().
    """
    return ntg.noisify(text)


def ngram(text):
    """Given a text, return the ngrams(3, 7)."""
    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
        
    ngrams = [grams
              for i in range(3, 7)
              for grams in find_ngrams(tokenizer.word_tokenize(text), i)]
    return ngrams


def normalize_subwords_accents(text):
    return ntg.normalize_accent_style_subwords(text)


def normalize_words_accents(text):
    return ntg.normalize_accent_style_words(text)


def accents_subword(text):
    return ntg.accent_style_subwords(text)


def accents_word(text):
    return ntg.accent_style_words(text)


def collect_dataset(src, tgt, max_seq_len=50,
                    char_level_emb=False, augment_data=False,
                    shuffle=False, size=10000000):
    """Generate a parallel clean and noisy text from a given clean text."""
    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        sentences = [content.strip()
                     for content in contents.splitlines()
                     if content]

    process_pool = Pool()
    dataset = []
    if shuffle:
        random.shuffle(sentences)

    dataset.extend(sentences[:size])
    dataset_size = len(dataset)
    print('  [+] Flattened data length: {}'.format(dataset_size))

    if augment_data:
        print('  [+] N-grams to augment the data.')
        ngrams = process_pool.map(ngram, dataset[:50000])
        print('    [-] Flattening n-grams...')
        dataset_ngrams = [' '.join(list(gram))
                          for grams in ngrams
                          for gram in grams
                          if gram]
        print('    [-] Shuffling n-grams')
        random.shuffle(dataset_ngrams)
        dataset.extend(dataset_ngrams[:size*5])
        print('    [-] Add ngrams to dataset')
        dataset_size = len(dataset)
        print('  [+] New dataset size: {}'.format(dataset_size))

    if shuffle:
        print('  [+] Randomizing the position of the dataset')
        random.shuffle(dataset)

    if size:
        dataset = dataset[:size]

    re_fix_punct = re.compile(r' (?=\W)')
    re_fix_punct_others = re.compile(r'([\[\]\(\)";:\']) (\w+)')
    
    detokenizer = MosesDetokenizer()

    rules = ['group_repeating_units',
             'phonetic_style',
             'remove_vowels',
             None]

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

            clean_sentence = sentence[:max_seq_len]
            clean_sentence = clean_sentence[:clean_sentence.rfind(' ')]


            # Normalize the contracted words from Articles
            clean_sentence = ntg.raw_daw.sub(ntg.normalize_raw_daw,
                                             clean_sentence)

            clean_sentence = ntg.expand_pattern.sub(ntg.expand_repl,
                                                    clean_sentence)


            noisy_sentence = clean_sentence
            noisy_sentence = ntg.mwe_tokenizer.tokenize(tokenizer.word_tokenize(noisy_sentence))

            noisy_sentence = process_pool.map(accents_word, noisy_sentence)
            # noisy_sentence = process_pool.map(accents_subword, noisy_sentence)
            noisy_sentence = ' '.join(noisy_sentence)
            noisy_sentence = ntg.raw_daw.sub(ntg.noisify_raw_daw,
                                             noisy_sentence)

            noisy_sentence = ntg.contract_pattern.sub(
                ntg.contract_repl, noisy_sentence)

            # for re_exp, repl in ntg.text_patterns:
            #     if random.getrandbits(1):
            #         noisy_sentence = re_exp.sub(repl, noisy_sentence)

            noisy_sentence = noisy_sentence.split()

            try:
                sos = ntg.noisify(noisy_sentence[0], sos=True)
                ntg.rule = random.choice(rules)
                noisy_sentence = process_pool.map(
                    noisify, noisy_sentence[1:])
                noisy_sentence.insert(0, sos)
            except IndexError:
                # It is faster than checking length of the list
                pass

            # noisy_sentence = ' '.join(noisy_sentence)

            clean_sentence = clean_sentence.split()

            clean_sentence = detokenizer.detokenize(clean_sentence, return_str=True) \
                .replace('``', '"') \
                .replace("''", '"') \
                .replace("( ", "(")

            noisy_sentence = detokenizer.detokenize(noisy_sentence, return_str=True) \
                .replace('``', '"') \
                .replace("''", '"') \
                .replace("( ", "(")

            if char_level_emb:
                clean_sentence = ' '.join(list(clean_sentence)) \
                                    .replace(' ' * 3, ' <space> ')
                noisy_sentence = ' '.join(list(noisy_sentence)) \
                                    .replace(' ' * 3, ' <space> ')

            if clean_sentence and noisy_sentence:
                decoder_file.write(clean_sentence + "\n")
                encoder_file.write(noisy_sentence + "\n")

        decoder_file.truncate(decoder_file.tell() - 2)
        encoder_file.truncate(encoder_file.tell() - 2)


accent_subwords_dict = csv_to_dict(os.path.join(
    'training', 'data', 'accented_subwords.dic'))

accent_words_dict = csv_to_dict(
    os.path.join('training', 'data', 'accented_words.dic'))

phonetic_subwords_dict = csv_to_dict(
    os.path.join('training', 'data', 'phonetically_styled_subwords.dic'))

phonetic_words_dict = csv_to_dict(
    os.path.join('training', 'data', 'phonetically_styled_words.dic'))

with open(os.path.join('training', 'data', 'hyph_fil.tex'), 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_subwords_dict=accent_subwords_dict,
                    accent_words_dict=accent_words_dict,
                    phonetic_subwords_dict=phonetic_subwords_dict,
                    phonetic_words_dict=phonetic_words_dict,
                    hyphenator_dict=hyphenator_dict)