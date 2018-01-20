import re
import random
import string
from nltk.tokenize import MWETokenizer
from training.data.hyphenator import Hyphenator


class TextNoisifier:
    def __init__(self, accent_dict, phonetic_style_dict,
                 contraction_dict, expansion_dict, hyphenator_dict):
        self.accent_dict = accent_dict
        self.phonetic_style_dict = phonetic_style_dict
        self.contraction_dict = contraction_dict
        self.expansion_dict = expansion_dict
        self.re_adj_vowels = re.compile(r'[aeiou]{2,}')
        self.re_digits = re.compile(r'-?\d+')
        self.re_accepted = re.compile(r"^[\sA-Za-z'-]+")
        self.re_hyphens = re.compile(r'(-)')
        self.misspell_replacement = list(string.ascii_lowercase) + ['']
        self.vowels = "aeiouAEIOU"

        matches = re.findall(r"\{(.*?)\}",
                             hyphenator_dict,
                             re.MULTILINE | re.DOTALL)
        patterns = matches[0]
        exceptions = matches[1]
        self.hyphenator = Hyphenator(patterns, exceptions)

        mwe = []
        for k, v in contraction_dict.items():
            words = k.split()
            if len(words) > 1:
                mwe.append((words[0], words[1]))

        self.mwe_tokenizer = MWETokenizer(mwe, separator=' ')

        self.amable_expr = re.compile(r'\b[aA]ng\s\b[bp]\w+\b')
        self.anable_expr = re.compile(r'\b[aA]ng\s\b[sldt]\w+\b')

        self.anu_expr = re.compile(r'\b[aA]no\s\b\w{2}\b')

        self.expandable_expr = re.compile(r"\w[aeiou]'[yt]$",
                                          re.IGNORECASE)
        self.contractable_expr = re.compile(r'\w+[aeiou]\sa[ty]\b',
                                            re.IGNORECASE)

        self.remove_space_expr = re.compile(r'\b(pa|na|di)\b\s\w+',
                                            re.IGNORECASE)

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
        letter = random.choice(list(self.vowels))
        length = random.randrange(4, 10)
        if random.getrandbits(1):
            word = word.replace(letter, letter * length, 1)
        else:
            word = word[::-1].replace(letter, letter * length, 1)[::-1]
        return word

    def misspell(self, word):
        letter = word[random.randrange(len(word))]
        replacement = random.choice(self.misspell_replacement)
        if random.getrandbits(1):
            word = word.replace(letter, replacement, 1)
        else:
            word = word[::-1].replace(letter, replacement, 1)[::-1]

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
        words = self.mwe_tokenizer.tokenize(text.split())
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

    def noisify(self, word):
        if not self.re_digits.search(word) \
                and self.re_accepted.search(word) \
                and len(word) > 1 \
                and word[0].islower() \
                and "'" not in word:

            grouped_units = self.group_repeating_units(word)
            if grouped_units != word:
                return grouped_units

            selected = random.choice(['remove_vowels',
                                      'phonetic_style',
                                      'accent_style'])
            noisy_word = self.dispatch_rules(selected, word)

            if noisy_word == word:
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

    @staticmethod
    def word_contraction(match):
        return match.group(0)[:-3] + "'" + match.group(0)[-1:]

    @staticmethod
    def word_ang_to_am(match):
        return match.group(0)[:1] + 'm' + match.group(0)[4:]

    @staticmethod
    def word_ang_to_an(match):
        return match.group(0)[:2] + match.group(0)[4:]

    @staticmethod
    def word_ano(match):
        return match.group(0)[:2] + 'u' + match.group(0)[4:]

    @staticmethod
    def word_expansion(match):
        return match.group(0)[:-2] + " a" + match.group(0)[-1:]

    @staticmethod
    def word_remove_space(match):
        return match.group(0).replace(' ', '')
