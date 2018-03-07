import random
import re
from pprint import pprint

import nltk.tokenize as tokenizer
from .hyphenator import Hyphenator


class TextNormalizer:
    """Used for normalization of common errors in the dataset."""
    def __init__(self,
                 accent_words_dict,
                 spell_corrector,
                 pandiwa_words_dict,
                 hyphenator_dict):

        self.spell_corrector = spell_corrector
        self.accent_words_dict = accent_words_dict

        self.vowels = 'aeiou'
        self.vowels += self.vowels.upper()
        self.consonants = "bcdfghjklmnpqrstvwxyz"
        self.consonants += self.consonants.upper()
        self.alphabet = self.vowels + self.consonants

        matches = re.findall(r"{(.*?)\}",
                             hyphenator_dict,
                             re.MULTILINE | re.DOTALL)
        patterns = matches[0]
        exceptions = matches[1]
        self.hyphenator = Hyphenator(patterns, exceptions)
        self.mwes = []
        for k, v in accent_words_dict.items():
            words = v.split()
            if len(words) > 1:
                self.mwes.append(tuple(words))
                words[0] = words[0].capitalize()
                self.mwes.append(tuple(words))

        print("============= Multi-word Expressions =================")
        pprint(self.mwes)

        self.mwe_tokenizer = tokenizer.MWETokenizer(self.mwes, separator=' ')

        self.expand_pattern = re.compile(r"(\w+[aeiou])'([yt])", re.IGNORECASE)
        self.expand_repl = r'\1 a\2'

        self.text_patterns = [
            # Ang baho -> Ambaho
            (re.compile(r'(\b[aA]m)(\b[bp]\w+\b)'), r'\1ng \2'),
            # Ang dami -> Andami
            (re.compile(r'(\b[aA]n)(\b[gkhsldt]\w+\b)'), r'\1g \2'),
            # Ano ba -> Anuba
            (re.compile(r'(\b[aA]n)(\b\w{,3}\b)'), r'\1o \2'),
            # Pagkain -> Pag kain
            (re.compile(r'(pag|mag|ika|kasing|maki|paki|labing)\s(\w+)'), r'\1\2'),
        ]

        self.common_prefix = ['mag', 'ika', 'maki', 'paki',
                              'pag', 'kasing', 'labing']
        self.raw_daw = \
            re.compile(r'\b([^aeiou]|[aeiou])\b\s([dr])(aw|ito|oon|in)',
                       re.IGNORECASE)

    @staticmethod
    def _format(match, repl):
        return "{} {}{}".format(
            match.group(1),
            repl if match.group(2).islower() else repl.upper(),
            match.group(3))

    def 

    def raw_daw_repl(self, match):
        """Normalize text that misuse raw and daw before noisification."""
        if match.group(1) in self.vowels:
            return self._format(match, 'r')  # raw
        elif match.group(1) in self.consonants:
            return self._format(match, 'd')  # daw

    def spell_correct(self, word):
        return self.spell_corrector.correction(word)

    def expand_expr(self, text):
        return self.expand_pattern.sub(self.expand_repl, text)

    def accent_style(self, text):
        return self._substitution(text, self.accent_words_dict)

    def ng2nang(self, text):
        pass

    def nang2ng(self, text):
        pass

    def hyphen_repeating_units(self, units):
        pass
    
    def hypen_vowel_rootword(self, word):
        pass

    @staticmethod
    def _substitution(word, substitution_dict):
        """Find the substitute in the text and replace."""
        is_upper = word[0].isupper()
        is_allcaps = str.isupper(word)

        word = word.lower()
        word = substitution_dict.get(word, word)
        word = word.replace("'", '')
        if is_upper:
            word = word.capitalize()
            if is_allcaps:
                word = word.upper()
        return word
