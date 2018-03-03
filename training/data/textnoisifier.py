"""This module handles all noisy text generation."""
import re
import random
from pprint import pprint
import nltk.tokenize as tokenizer
from .hyphenator import Hyphenator


class TextNoisifier:
    """Handles all text noisification rules."""

    def __init__(self,
                 accent_words_dict,
                 phonetic_subwords_dict,
                 phonetic_words_dict,
                 hyphenator_dict):
        """Initialize all dictionaries and Regex for string manipulation."""
        self.accent_words_dict = accent_words_dict

        self.phonetic_words_dict = phonetic_words_dict
        self.phonetic_subwords_dict = phonetic_subwords_dict

        self.re_adj_vowels = re.compile(r'[aeiou]{2,}', re.IGNORECASE)
        self.re_accepted = re.compile(r"\b[\sA-Za-z'-]+\b")
        self.re_hyphens = re.compile(r'(-)')

        self.vowels = 'aeiou'
        self.lower_vowels = self.vowels
        self.upper_vowels = self.vowels.upper()
        self.vowels += self.upper_vowels

        self.consonants = "bcdfghjklmnpqrstvwxyz"
        self.lower_consonants = self.consonants
        self.upper_consonants = self.consonants.upper()
        self.consonants += self.upper_consonants

        self.alphabet = self.vowels + self.consonants
        self.lower_alphabet = ''.join(list(set(self.alphabet.lower())))
        self.upper_alphabet = ''.join(list(set(self.alphabet.upper())))

        matches = re.findall(r"{(.*?)\}",
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

        """
        >>> nang
        Replacement for 'noong'.
        Sagot sa paano o gaano.
        Pang-angkop sa pandiwang inuulit (Tulog ka nang tulog)
        Kung pinagsamang 'na' at 'ang', 'na' at 'ng', o ng 'na' at 'na'

        >>> ng
        Pantukoy ng pangalan
        Pagpapahiwatig ng pagmamay-ari
        """
        # TODO: Implement this randomly and see the result if it works.
        self.ng2nang_pattern = re.compile(r'\bng\b')
        self.ng2nang_repl = r'nang'
        self.nang2ng_pattern = re.compile(r'r\bnang\b')
        self.nang2ng_repl = r'ng'

        self.raw_daw = \
            re.compile(r'\b([^aeiou]|[aeiou])\b\s([dr])(aw|ito|oon|in)',
                       re.IGNORECASE)

    def ng2nang(self, text):
        return self.ng2nang_pattern.sub(self.ng2nang_repl, text)

    def nang2ng(self, text):
        return self.nang2ng_pattern.sub(self.nang2ng_repl, text)

    def contract_expr(self, text):
        return self.contract_pattern.sub(self.contract_repl, text)

    @staticmethod
    def _format(match, repl):
        return "{} {}{}".format(
            match.group(1),
            repl if match.group(2).islower() else repl.upper(),
            match.group(3))

    def raw_daw_repl(self, match):
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
        letter = random.choice(self.alphabet)
        length = random.randrange(2, 10)
        if random.getrandbits(1):
            word = word.replace(letter, letter * length, 1)
        else:
            word = word[::-1].replace(letter, letter * length, 1)[::-1]
        return word

    def misspell(self, word):
        """Replace/Delete/Insert a character in a word to misspell."""
        if random.getrandbits(1):
            word = self._one_char_edit(word)
        else:
            word = self._two_char_edit(word)
        return word

    def _one_char_edit(self, word):
        edit = random.choice(['del', 'ins', 'rep', 'tra'])
        idx = random.randrange(len(word) + 1)
        letter = random.choice(self.lower_alphabet)

        lsplit, rsplit = word[:idx], word[idx:]
        if edit == 'del' and rsplit:
            word = lsplit + rsplit[1:] 
        elif edit == 'tra' and len(rsplit) > 1:
            word = lsplit + rsplit[1] + rsplit[0] + rsplit[2:]
        elif edit == 'rep':
            word = lsplit + letter + rsplit[1:]
        elif edit == 'ins':
            word = lsplit + letter + rsplit
        return word

    def _two_char_edit(self, word):
        return self._one_char_edit(self._one_char_edit(word))

    def phonetic_style_subwords(self, word):
        """Return a phonetically styled portion of a word."""
        return self._subword_substitution(word, self.phonetic_subwords_dict)

    def phonetic_style_words(self, word):
        """Return a phonetically styled word."""
        return self._word_substitution(word, self.phonetic_words_dict)

    @staticmethod
    def _word_substitution(word, substitution_dict):
        """Find the substitute in the text and replace."""
        is_upper = word[0].isupper()
        is_allcaps = str.isupper(word)

        word = word.lower()
        v = substitution_dict.get(word, word)
        repl = random.choice(v) if isinstance(v, list) else v
        word = word.replace(word, repl)
        word = word.replace("'", '')
        if is_upper:
            word = word.capitalize()
            if is_allcaps:
                word = word.upper()
        return word

    @staticmethod
    def _subword_substitution(word, substitution_dict):
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
        text = self._word_substitution(text, self.accent_words_dict)
        return self._subword_substitution(text, self.accent_words_dict)

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
        """Randomly apply string manipulation.

        It only accepts alphabet, apostrophe and hyphen. 
        The string length must be greater than 1.
        It doesn't accept capital letters, hacky way to avoid named-entity to be noisified.

        """
        if self.re_accepted.search(word) \
                and len(word) > 1 \
                and (sos or word[0].islower()):
            
            value = random.random()
            if value > 0.90:  # slim chance
                selected = 'repeat_characters'
            elif value > 0.3:
                selected = 'misspell'
            else:
                selected = random.choice(['remove_vowels',
                                          'phonetic_style',
                                          'group_repeating_units',
                                          'accent_style'])

            word = self.dispatch_rules(selected, word)
            word = word.replace('-', '')
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
