import re
import random
import string
from utils.tokenizer import mwe_tokenize
from .hyphenator import Hyphenator


class TextNoisifier:
    def __init__(self, accent_dict, phonetic_style_dict,
                 contraction_dict, expansion_dict, hyphenator_dict):
        self.accent_dict = accent_dict
        self.phonetic_style_dict = phonetic_style_dict
        self.contraction_dict = contraction_dict
        self.expansion_dict = expansion_dict
        self.re_adj_vowels = re.compile(r'[aeiou]{2,}', re.IGNORECASE)
        self.re_accepted = re.compile(r"\b[\sA-Za-z'-]+\b")
        self.re_hyphens = re.compile(r'(-)')
        self.vowels = 'aeiou'
        self.vowels += self.vowels.upper()
        self.consonants = "bcdfghjklmnpqrstvwxyz"
        self.consonants += self.consonants.upper()

        self.rules = ['remove_vowels',
                      'phonetic_style',
                      'group_repeating_units',
                      'accent_style',
                      'repeat_characters',
                      'misspell']

        matches = re.findall(r"\{(.*?)\}",
                             hyphenator_dict,
                             re.MULTILINE | re.DOTALL)
        patterns = matches[0]
        exceptions = matches[1]
        self.hyphenator = Hyphenator(patterns, exceptions)

        self.mwes = []
        for k, v in contraction_dict.items():
            words = k.split()
            if len(words) > 1:
                self.mwes.append(tuple(words))
        self.expand_pattern = re.compile(r"(\w+[aeiou])'([yt])", re.IGNORECASE)
        self.expand_repl = r'\1 a\2'

        self.contract_pattern = re.compile(r'(\w+[aeiou])\s\ba([ty]\b)', re.IGNORECASE)
        self.contract_repl = r"\1'\2"

        self.text_patterns = [
            (re.compile(r'(\b[aA])(ng\s)(\b[bp]\w+\b)'), r'\1m\3'),  # ang baho -> ambaho
            (re.compile(r'(\b[aA]n)(g\s)(\b[gkhsldt]\w+\b)'), r'\1\3'),  # ang gulo -> angulo
            (re.compile(r'(\b[aA]n)(o\s)(\b\w{2}\b)'), r'\1u\3'),  # ano ba -> anuba
            (re.compile(r'(\b(ka|pa|na|di)\b)\s(\b[A-Za-z]{,5}\b)', re.IGNORECASE), r'\1\3')  # ka na -> kana
        ]

        self.raw_daw = re.compile(r'([^aeiou]|[aeiou])\b\s(d|r)(aw|ito|oon|in)', re.IGNORECASE)

    @staticmethod
    def _format(match, repl):
        return "{} {}{}".format(
            match.group(1),
            repl if match.group(2).islower() else repl.upper(),
            match.group(3))

    def normalize_raw_daw(self, match):
        if match.group(1) in self.vowels:
            return self._format(match, 'r')  # raw
        elif match.group(1) in self.consonants:
            return self._format(match, 'd')  # daw
        
    def noisify_raw_daw(self, match):
        if match.group(1) in self.vowels:
            return self._format(match, 'd')  # raw
        elif match.group(1) in self.consonants:
            return self._format(match, 'r')  # daw

    def remove_vowels(self, word):
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
        letter = random.choice(list(word))
        length = random.randrange(4, 10)
        if random.getrandbits(1):
            word = word.replace(letter, letter * length, 1)
        else:
            word = word[::-1].replace(letter, letter * length, 1)[::-1]
        return word

    def misspell(self, word):
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

    def phonetic_style(self, word):
        for k, v in self.phonetic_style_dict.items():
            word = word.replace(k, v)
        return word

    def accent_style(self, word):
        for k, v in self.accent_dict.items():
            if random.getrandbits(1):
                word = word.replace(k, v, 1)  # replace from left
            else:
                word = word[::-1].replace(k, v, 1)[::-1]  # replace from right
        return word

    def _dict_substitution(self, text, substitution_dict):
        words = mwe_tokenize(text, self.mwes)
        for i in range(len(words)):
            try:
                words[i] = substitution_dict[words[i]]
                words[i] = words[i].replace("'", '')
            except KeyError:
                pass
        return ' '.join(words)

    def contraction(self, text):
        return self._dict_substitution(text, self.contraction_dict)

    def expansion(self, text):
        return self._dict_substitution(text, self.expansion_dict)

    def group_repeating_units(self, word):
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
        for i in range(len(units) - 1):
            if units[i] != '' and units[i].lower() == units[i + 1].lower():
                units[i + 1] = str(2)
            elif units[i] != '' and units[i].lower() \
                    == units[i + 1][:-(len(units[i]))].lower():
                units[i + 1] = str(2) + units[i + 1][(len(units[i])):]
        return ''.join(units)

    def noisify(self, word, sos=False):
        if self.re_accepted.search(word) \
                and len(word) > 1 \
                and "'" not in word:

            selected = random.choice(['remove_vowels',
                                      'phonetic_style',
                                      'group_repeating_units',
                                      'accent_style'])

            noisy_word = self.dispatch_rules(selected, word)

            if '-' in noisy_word:
                word = noisy_word.replace('-', '')
            elif noisy_word == word:
                selected = random.choice(['repeat_characters'])
                word = self.dispatch_rules(selected, word)
            else:
                word = noisy_word

        return word

    def dispatch_rules(self, rule, word):
        return {
            'remove_vowels': self.remove_vowels(word),
            'phonetic_style': self.phonetic_style(word),
            'accent_style': self.accent_style(word),
            'repeat_characters': self.repeat_characters(word),
            'misspell': self.misspell(word),
            'group_repeating_units': self.group_repeating_units(word)
        }.get(rule, word)