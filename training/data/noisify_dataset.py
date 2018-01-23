"""Module that generates the dataset."""
import os
import random
import time
from multiprocessing import Pool
from .textnoisifier import TextNoisifier


def csv_to_dict(file):
    """Convert csv to python dictionary."""
    d = {}
    with open(file, 'r') as f:
        rows = f.read().split('\n')
        for row in rows:
            k, v = row.split(',')
            d.update({k: v})
    return d


def is_ascii(text):
    """ASCII Characters only."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


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
              for grams in find_ngrams(text.split(), i)
              ]
    return ngrams


def collect_dataset(src, tgt, max_seq_len=50,
                    char_level_emb=False, augment_data=False,
                    shuffle=False, size=None):
    """Generate a parallel clean and noisy text from a given clean text."""
    process_pool = Pool()

    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        sentences = [content.strip()
                     for content in contents.split('\n')
                     if content]

    dataset = []
    if shuffle:
        sentences = random.sample(sentences, len(sentences))

    dataset.extend(sentences)
    dataset_size = len(dataset)
    print('  [+] Flattened data length: {}'.format(dataset_size))

    if augment_data:
        ngrams = process_pool.map(ngram, dataset)
        dataset_ngrams = [' '.join(list(gram))
                          for grams in ngrams
                          for gram in grams
                          if gram]

        dataset_ngrams = random.sample(dataset_ngrams, len(dataset_ngrams))
        dataset.extend(dataset_ngrams[:200000])
        print('  [+] Add ngrams to dataset')
        dataset_size = len(dataset)
        print('  [+] New dataset size: {}'.format(dataset_size))

    if shuffle:
        print('  [+] randomizing the position of the dataset')
        dataset = random.sample(dataset, len(dataset))

    if size:
        dataset = dataset[:size]

    sent_number = 0
    start_time = time.time()

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

            # Normalize first the contracted words from News Site Articles
            clean_sentence = ntg.expansion(clean_sentence)
            clean_sentence = ntg.expand_pattern.sub(ntg.expand_repl,
                                                    clean_sentence)

            noisy_sentence = clean_sentence

            if random.getrandbits(1):
                noisy_sentence = ntg.contraction(noisy_sentence)

            noisy_sentence = ntg.contract_pattern.sub(
                ntg.contract_repl, noisy_sentence)

            for re_exp, repl in ntg.text_patterns:
                noisy_sentence = re_exp.sub(repl, noisy_sentence)

            noisy_sentence = noisy_sentence.split()

            sos = ntg.noisify(noisy_sentence[0], sos=True)

            noisy_sentence = process_pool.map(
                noisify, noisy_sentence[1:])

            noisy_sentence.insert(0, sos)

            noisy_sentence = ' '.join(noisy_sentence)

            if char_level_emb:
                clean_sentence = ' '.join(list(clean_sentence)) \
                                    .replace(' ' * 3, ' <space> ')
                noisy_sentence = ' '.join(list(noisy_sentence)) \
                                    .replace(' ' * 3, ' <space> ')

            if clean_sentence and noisy_sentence:
                decoder_file.write(clean_sentence + "\n")
                encoder_file.write(noisy_sentence + "\n")

        decoder_file.truncate(decoder_file.tell() - 1)
        encoder_file.truncate(encoder_file.tell() - 1)


accent_dict = csv_to_dict(os.path.join(
    'training', 'data', 'common_accented_words.dic'))

contract_dict = csv_to_dict(
    os.path.join('training', 'data', 'common_contracted_words.dic'))

phonetic_dict = csv_to_dict(
    os.path.join('training', 'data', 'common_phonetically_styled_words.dic'))

expansion_dict = {v: k for k, v in contract_dict.items()}

with open(os.path.join('training', 'data', 'hyph_fil.tex'), 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_dict, phonetic_dict, contract_dict,
                    expansion_dict, hyphenator_dict)