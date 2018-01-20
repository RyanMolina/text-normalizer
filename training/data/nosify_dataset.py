import os
import random
import time
from multiprocessing import Pool
import nltk
from training.data.textnoisifier import TextNoisifier


def csv_to_dict(file):
    d = {}
    with open(file, 'r') as f:
        rows = f.read().split('\n')
        for row in rows:
            k, v = row.split(',')
            d.update({k: v})
    return d


def is_ascii(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def noisify(text):
    """
    function wrapper to be fed on Pool.map()
    :param text:
    :return noisy text:
    """
    return ntg.noisify(text)


def ngram_wrapper(text):
    ngrams = [grams
              for i in range(6)
              for grams in nltk.ngrams(text.split(), i)
              ]
    return ngrams


def space_seperated(text):
    return ' '.join(nltk.word_tokenize(text))


def collect_dataset(src, tgt, tok=None, max_seq_len=50,
                    char_level_emb=False, augment_data=False,
                    shuffle=False, size=None):

    process_pool = Pool()

    # Instance of PunktSentenceTokenizer from nltk.tokenize module
    if tok:
        tokenizer = nltk.data.load(tok).tokenize
    else:
        tokenizer = nltk.sent_tokenize

    print('# Reading file')
    with open(src, encoding='utf8') as infile:
        contents = infile.read()
        articles = [content.strip()
                    for content in contents.split('\n')
                    if content != '\n']

    dataset = []

    if shuffle:
        articles = random.sample(articles, len(articles))

    print('  [+] converting to list of sentences')
    articles_sentences = process_pool.map(tokenizer, articles)
    print('  [+] flattening the list of sentences')
    sentences = [sentence
                 for sentences in articles_sentences
                 for sentence in sentences]
    dataset.extend(sentences)
    dataset_size = len(dataset)
    print('  [+] Flattened data length: {}'.format(dataset_size))

    if augment_data:
        ngrams = process_pool.map(ngram_wrapper, dataset)
        dataset_ngrams = [' '.join(list(gram))
                          for grams in ngrams
                          for gram in grams
                          if gram]

        dataset_ngrams = random.sample(dataset_ngrams, len(dataset_ngrams))
        dataset.extend(dataset_ngrams[:500000])
        print('  [+] Add ngrams to dataset')
        dataset_size = len(dataset)
        print('  [+] New dataset size: {}'.format(dataset_size))

    if not char_level_emb:
        dataset = process_pool.map(space_seperated, dataset)

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
            clean_sentence = ntg.expandable_expr.sub(ntg.word_expansion,
                                                     clean_sentence)

            noisy_sentence = clean_sentence

            noisy_sentence = ntg.contraction(noisy_sentence)

            noisy_sentence = ntg.contractable_expr.sub(
                ntg.word_contraction, noisy_sentence)

            noisy_sentence = ntg.anable_expr.sub(
                ntg.word_ang_to_an, noisy_sentence)

            noisy_sentence = ntg.anu_expr.sub(
                ntg.word_ano, noisy_sentence)

            noisy_sentence = ntg.amable_expr.sub(
                ntg.word_ang_to_am, noisy_sentence)

            noisy_sentence = ntg.remove_space_expr.sub(
                ntg.word_remove_space, noisy_sentence)

            noisy_sentence = ' '.join(process_pool.map(
                noisify, noisy_sentence.split()))

            # if random.getrandbits(1):
            #     puncts = "!.?,:;"
            #     punct_sample = random.sample(
            #         puncts, random.randrange(len(puncts)))
            #     remove_punct_rule = \
            #         str.maketrans(dict.fromkeys(punct_sample, None))
            #     noisy_sentence = noisy_sentence.translate(remove_punct_rule)

            if char_level_emb:
                clean_sentence = ' '.join(list(clean_sentence)) \
                                    .replace(' ' * 3, ' <space> ')
                noisy_sentence = ' '.join(list(noisy_sentence)) \
                                    .replace(' ' * 3, ' <space> ')

            if clean_sentence and noisy_sentence:
                print(clean_sentence, file=decoder_file)
                print(noisy_sentence, file=encoder_file)

        decoder_file.truncate(decoder_file.tell() - 1)
        encoder_file.truncate(encoder_file.tell() - 1)


accent_dict = csv_to_dict(os.path.join(
    'training', 'data', 'common_accented_words.txt'))

contract_dict = csv_to_dict(
    os.path.join('training', 'data', 'common_contracted_words.txt'))

phonetic_dict = csv_to_dict(
    os.path.join('training', 'data', 'common_phonetically_styled_words.txt'))

expansion_dict = {v: k for k, v in contract_dict.items()}

with open(os.path.join('training', 'data', 'hyph_fil.tex'), 'r') as f:
    hyphenator_dict = f.read()

ntg = TextNoisifier(accent_dict, phonetic_dict, contract_dict,
                    expansion_dict, hyphenator_dict)
