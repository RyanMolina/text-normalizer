"""This module handles all noisy text generation."""
import re
import random
import string
from pprint import pprint
# from utils.tokenizer import mwe_tokenize
import nltk.tokenize as tokenizer
from .hyphenator import Hyphenator


class TextNoisifier:
    """Handles all text noisification rules."""

    def __init__(self, accent_words_dict,
                 phonetic_subwords_dict, phonetic_words_dict, hyphenator_dict):
        """Initialize all dictionaries and Regex for string manipulation."""
        self.accent_words_dict = accent_words_dict

        self.phonetic_words_dict = phonetic_words_dict
        self.phonetic_subwords_dict = phonetic_subwords_dict

        self.re_adj_vowels = re.compile(r'[aeiou]{2,}', re.IGNORECASE)
        self.re_accepted = re.compile(r"\b[\sA-Za-z'-]+\b")
        self.re_hyphens = re.compile(r'(-)')

        self.vowels = 'aeiou'
        self.vowels += self.vowels.upper()
        self.consonants = "bcdfghjklmnpqrstvwxyz"
        self.consonants += self.consonants.upper()

        matches = re.findall(r"\{(.*?)\}",
                             hyphenator_dict,
                             re.MULTILINE | re.DOTALL)

        patterns = matches[0]
        exceptions = matches[1]
        self.hyphenator = Hyphenator(patterns, exceptions)

        self.mwes = []
        for k, v in accent_words_dict.items():
            words = k.split()
            if len(words) > 1:
                self.mwes.append(tuple(words))
                words[0] = words[0].capitalize()
                self.mwes.append(tuple(words))
        print("============= Multi-word Expressions =================")
        pprint(self.mwes)

        self.mwe_tokenizer = tokenizer.MWETokenizer(self.mwes)
        self.expand_pattern = re.compile(r"(\w+[aeiou])'([yt])", re.IGNORECASE)
        self.expand_repl = r'\1 a\2'

        self.contract_pattern = re.compile(r'(\w+[aeiou])\s\ba([ty]\b)',
                                           re.IGNORECASE)
        self.contract_repl = r"\1'\2"

        # FIXME: Has the tendency to normalize words like "Antipolo", "Antik"
        self.text_patterns = [
            # Ang baho -> Ambaho
            (re.compile(r'(\b[aA])(ng\s)(\b[bp]\w+\b)'), r'\1m\3'),
            # Ang dami -> Andami
            (re.compile(r'(\b[aA]n)(g\s)(\b[gkhsldt]\w+\b)'), r'\1\3'),
            # Ano ba -> Anuba
            (re.compile(r'(\b[aA]n)(os)(\b\w{2}\b)'), r'\1u\3'),
            # Pagkain -> Pag kain
            (re.compile(r'(pag)(\w+)'), r'\1 \2'),
            # na naman -> nanaman
            (re.compile(r'(\b(ka|pa|na|di)\b)\s(\b[A-Za-z]{,2}\b)',
             re.IGNORECASE), r'\1\3')
        ]

        # TODO: Implement this randomly and see the result if it works.
        self.ng_nang = [
            (re.compile(r'\bng\b'), r'nang'),
            (re.compile(r'r\bnang\b'), r'ng'),
        ]

        self.raw_daw = \
            re.compile(r'\b([^aeiou]|[aeiou])\b\s(d|r)(aw|ito|oon|in)',
                       re.IGNORECASE)

    @staticmethod
    def _format(match, repl):
        return "{} {}{}".format(
            match.group(1),
            repl if match.group(2).islower() else repl.upper(),
            match.group(3))

    def normalize_raw_daw(self, match):
        """Normalize text that misuse raw and daw."""
        if match.group(1) in self.vowels:
            return self._format(match, 'r')  # raw
        elif match.group(1) in self.consonants:
            return self._format(match, 'd')  # daw

    def noisify_raw_daw(self, match):
        """Misuse raw and daw in sentence."""
        if match.group(1) in self.vowels:
            return self._format(match, 'd')  # raw
        elif match.group(1) in self.consonants:
            return self._format(match, 'r')  # daw

    def remove_vowels(self, word):
        """Remove vowels randomly from a word."""
        vowel_sample = random.sample(self.vowels,
                                     random.randrange(len(self.vowels)))
        remove_vowel_rule = str.maketrans(dict.fromkeys(vowel_sample,
                                                        None))
        if len(word) == 4 and word[0] in self.vowels:
            if random.getrandbits(1):
                return word[1:]
        if not self.re_adj_vowels.search(word) and len(word) > 3:
            w_len = len(word)
            center = w_len // 2
            if random.getrandbits(1):
                if random.getrandbits(1):  # left
                    word = word[0] \
                        + word[1:center].translate(remove_vowel_rule) \
                        + word[center:]
                if random.getrandbits(1):  # middle
                    start = center // 2
                    end = center + start
                    word = word[:start] \
                        + word[start:end].translate(remove_vowel_rule) \
                        + word[end:]
                if random.getrandbits(1):  # right
                    word = word[:center] \
                        + word[center:w_len-1].translate(remove_vowel_rule) \
                        + word[-1]
            else:  # all
                word = word[0] \
                    + word[1:-1].translate(remove_vowel_rule) \
                    + word[-1]
        elif len(word) == 2 and word[-1] in self.vowels:
            word = word[0]
        return word

    def repeat_characters(self, word):
        """Repeat characters from left or right portion of the word."""
        letter = random.choice(list(word))
        length = random.randrange(4, 10)
        if random.getrandbits(1):
            word = word.replace(letter, letter * length, 1)
        else:
            word = word[::-1].replace(letter, letter * length, 1)[::-1]
        return word

    def misspell(self, word):
        """Replace/Delete/Insert a character in a word to misspell."""
        selected = random.choice(['replace', 'delete', 'insert'])
        letter = random.choice(list(word))
        if selected == 'replace':
            replacement = random.choice(string.ascii_lowercase)
            if random.getrandbits(1):
                word = word.replace(letter, replacement, 1)
            else:
                word = word[::-1].replace(letter, replacement, 1)[::-1]
        else:
            positions = [i for i, e in enumerate(list(word)) if e == letter]
            idx = random.choice(positions)
            if selected == 'delete':
                word = word[:idx] + word[idx + 1:]
            else:
                insert = random.choice(string.ascii_lowercase)
                word = word[:idx] + insert + word[idx:]
        return word

    def phonetic_style_subwords(self, word):
        """Return a phonetically styled portion of a word."""
        return self._subword_substitution(word, self.phonetic_subwords_dict)

    def phonetic_style_words(self, word):
        """Return a phonetically styled word."""
        return self._subword_substitution(word, self.phonetic_words_dict)

    @staticmethod
    def _substitution(word, substitution_dict):
        """Find the substitute in the text and replace."""
        for k, v in substitution_dict.items():
            try:
                is_upper = word[0].isupper()
                is_allcaps = str.isupper(word)
            except IndexError:
                continue
            word = word.lower()
            repl = random.choice(v) if isinstance(v, list) else v
            word = word.replace(k, repl)
            word = word.replace("'", '')
            if is_upper:
                word = word.capitalize()
                if is_allcaps:
                    word = word.upper()
        return word

    def phonetic_style(self, text):
        """Phonetic style for the word."""
        result = self.phonetic_style_words(text)
        return self.phonetic_style_subwords(result)

    def accent_style(self, text):
        """Accent style a word."""
        return self._substitution(text, self.accent_words_dict)

    def group_repeating_units(self, word):
        """Group repeating units by grouping the syllables."""
        hyphenated_words = self.re_hyphens.split(word)
        if len(hyphenated_words) > 1:
            if hyphenated_words[0].lower().find(hyphenated_words[2]) != -1:
                end = len(hyphenated_words[0])
                word = hyphenated_words[0][:end] \
                    + '2' + hyphenated_words[2][end:]
            elif hyphenated_words[2].find(hyphenated_words[0].lower()) != -1:
                end = len(hyphenated_words[0])
                word = hyphenated_words[0][:end] \
                    + '2' + hyphenated_words[2][end:]

        word = self.group_units(self.hyphenator.hyphenate_word(word))
        return word

    @staticmethod
    def group_units(units):
        """Group syllables with the no. of occurrences."""
        for i in range(len(units) - 1):
            if units[i] != '' and units[i].lower() == units[i + 1].lower():
                units[i + 1] = str(2)
            elif units[i] != '' and units[i].lower() \
                    == units[i + 1][:-(len(units[i]))].lower():
                units[i + 1] = str(2) + units[i + 1][(len(units[i])):]
        return ''.join(units)

    def noisify(self, word, sos=False):
        """Randomly apply string manipulation."""
        if self.re_accepted.search(word) \
                and len(word) > 1 \
                and (sos or word[0].islower()) \
                and "'" not in word:

            if random.random() > 0.95:  # slim chance
                selected = 'repeat_characters'
            elif random.random() > 0.5:
                selected = 'misspell'
            else:
                selected = random.choice(['remove_vowels',
                                          'phonetic_style',
                                          'group_repeating_units',
                                          'accent_style'])

            word = self.dispatch_rules(selected, word)

        return word

    def dispatch_rules(self, rule, word):
        """Text noisifier dispatcher."""
        return {
            'remove_vowels': self.remove_vowels(word),
            'phonetic_style': self.phonetic_style(word),
            'misspell': self.misspell(word),
            'accent_style': self.accent_style(word),
            'repeat_characters': self.repeat_characters(word),
            'group_repeating_units': self.group_repeating_units(word)
        }.get(rule, word)
