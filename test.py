import os
from pprint import pprint
from nltk.tokenize import word_tokenize
try:
    from .training.data.textnormalizer import TextNormalizer
    from .training.data.spellcorrector import SpellCorrector
    from .training.data.textnoisifier import TextNoisifier
    from .utils.helper import csv_to_dict
except SystemError:
    from training.data.textnormalizer import TextNormalizer
    from training.data.textnoisifier import TextNoisifier
    from training.data.spellcorrector import SpellCorrector
    from utils.helper import csv_to_dict

FILE_PATH = os.path.dirname(__file__)

accent_words_dict = csv_to_dict(
    os.path.join(FILE_PATH, 'training', 'data','accented_words.dic'))


phonetic_subwords_dict = csv_to_dict(
    os.path.join(FILE_PATH, 'training', 'data', 'phonetically_styled_subwords.dic'))

phonetic_words_dict = csv_to_dict(
    os.path.join(FILE_PATH, 'training', 'data', 'phonetically_styled_words.dic'))


accent_words_dict = {v2: k for k, v in accent_words_dict.items()
                           for v2 in v }

pprint(accent_words_dict)

dict_path = os.path.join(FILE_PATH, 'training', 'data', 'corpus', 'tagalog_sent.txt')

with open(os.path.join(FILE_PATH, 'training', 'data', 'hyph_fil.tex'), 'r') as f:
    hyphenator_dict = f.read()

spell_corrector = SpellCorrector(dict_path=dict_path)
tn = TextNormalizer(accent_words_dict=accent_words_dict,
                    hyphenator_dict=hyphenator_dict,
                    spell_corrector=spell_corrector)

with open(os.path.join(FILE_PATH, 'tagalog_sent.txt'), 'r') as in_fp:
    lines = in_fp.read().splitlines()
    with open(os.path.join(FILE_PATH, 'tagalog_sent_v2.txt'), 'w') as out_fp:
        for line in lines[:10]:
            line = tn.accent_style(line)
            print(line)
