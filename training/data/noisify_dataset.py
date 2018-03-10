"""Module that generates the dataset."""
import os
import random
import time
from multiprocessing import Pool
from .textnoisifier import TextNoisifier
from ...utils.helper import csv_to_dict
from .dataset_manipulation import manipulate


def noisify(word):
    """Given a text, return noisy text.

    This function also wraps the instance method to be passed on Pool map.
    Since, multiprocessing.Pool.map() only accept single argument function.
    This function hides the 'self' in ntg.noisify().
    """
    return ntg.noisify(word)


def noisify2(word):
    return ntg.noisify2(word)


def accent_style(word):
    return ntg.accent_style(word)


def phonetic_style(word):
    return ntg.phonetic_style(word)


def _generate(text):
    # Normalize the "r|d(oon, in, aw, ito mistakes" from Articles
    clean_sentence = text
    clean_sentence = ntg.raw_daw.sub(ntg.raw_daw_repl,
                                     clean_sentence)

    # Normalize the contracted words (Siya ay -> Siya'y) from Articles
    clean_sentence = ntg.expand_pattern.sub(ntg.expand_repl,
                                            clean_sentence)

    noisy_sentence = clean_sentence

    #  Contraction "Siya ay -> Siya'y"
    noisy_sentence = ntg.contract_pattern.sub(
        ntg.contract_repl, noisy_sentence)

    #  Noisify r|d(oon, in, aw, ito)
    noisy_sentence = ntg.raw_daw.sub(ntg.raw_daw_repl,
                                     noisy_sentence)

    # Misuse of word 'ng'
    noisy_sentence = ntg.nang2ng(noisy_sentence)

    # Split
    noisy_sentence = noisy_sentence.split()

    try:
        # 1st pass
        sos = ntg.noisify(noisy_sentence[0], sos=True)
        noisy_sentence = process_pool.map(noisify,
                                          noisy_sentence[1:])
        noisy_sentence.insert(0, sos)
        # 2nd pass
        sos = ntg.noisify(noisy_sentence[0], sos=True)
        noisy_sentence = process_pool.map(noisify2,
                                          noisy_sentence[1:])
        noisy_sentence.insert(0, sos)
    except IndexError:
        pass

    noisy_sentence = ' '.join(noisy_sentence)
    return clean_sentence, noisy_sentence


def run(src, tgt, max_seq_len=50,
        char_level_emb=False,
        augment_data=False, shuffle=False, size=None):

    dataset = manipulate(src,
                         shuffle=shuffle,
                         augment_data=augment_data,
                         size=size)
    dataset_size = len(dataset)
    sent_number = 0
    start_time = time.time()

    decoder_path = os.path.join(tgt, 'train.dec')
    encoder_path = os.path.join(tgt, 'train.enc')

    with open(decoder_path, 'w') as decoder_file, \
            open(encoder_path, 'w') as encoder_file:
        for sentence in dataset[:-500]:
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

            if max_seq_len:
                clean_sentence = sentence[:max_seq_len]
                clean_sentence = clean_sentence[:clean_sentence.rfind(' ')]
            else:
                clean_sentence = sentence

            clean_sentence, noisy_sentence = _generate(clean_sentence)

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

    accent_path = os.path.join(tgt, 'accent_style.dic')
    contraction_path = os.path.join(tgt, 'contractions.dic')
    misspelling_path = os.path.join(tgt, 'misspelling.dic')
    repeating_chars_path = os.path.join(tgt, 'repeating_characters.dic')
    repeating_units_path = os.path.join(tgt, 'repeating_units.dic')
    phonetic_path = os.path.join(tgt, 'phonetic_style.dic')

    with open(os.path.join(tgt, 'test.dec'), 'w') as decoder_file, \
            open(os.path.join(tgt, 'test.enc'), 'w') as encoder_file, \
            open(accent_path, 'w') as accent_file, \
            open(contraction_path, 'w') as contraction_file, \
            open(misspelling_path, 'w') as misspelling_file, \
            open(phonetic_path, 'w') as phonetic_file, \
            open(repeating_chars_path, 'w') as repeating_chars_file, \
            open(repeating_units_path, 'w') as repeating_units_file:

        for sentence in dataset[-500:]:
            if max_seq_len:
                clean_sentence = sentence[:max_seq_len]
                clean_sentence = clean_sentence[:clean_sentence.rfind(' ')]
            else:
                clean_sentence = sentence

            clean_sentence = ntg.raw_daw.sub(ntg.raw_daw_repl,
                                            clean_sentence)

            # Normalize the contracted words (Siya ay -> Siya'y) from Articles
            clean_sentence = ntg.expand_pattern.sub(ntg.expand_repl,
                                                    clean_sentence)
            noisy_sentence = [ntg.noisify(word, with_tag=True)
                              for word in ntg.mwe_tokenizer.tokenize(
                                  clean_sentence.split())]
            print(noisy_sentence)
            noisy_sent_output = ''
            for noisy_word, tag in noisy_sentence:
                if tag == 'accent_styles':
                    print(noisy_word, file=accent_file)
                elif tag == 'phonetic_styles':
                    print(noisy_word, file=phonetic_file)
                elif tag == 'contractions':
                    print(noisy_word, file=contraction_file)
                elif tag == 'misspellings':
                    print(noisy_word, file=misspelling_file)
                elif tag == 'repeating_characters':
                    print(noisy_word, file=repeating_chars_file)
                elif tag == 'repeating_units':
                    print(noisy_word, file=repeating_units_file)
                noisy_sent_output += noisy_word + ' '
            print(clean_sentence, file=decoder_file)
            print(noisy_sent_output, file=encoder_file)


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

process_pool = Pool()
