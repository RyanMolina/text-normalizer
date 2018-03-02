import re
from pprint import pprint

import nltk.tokenize as tokenizer
from .hyphenator import Hyphenator


class TextNormalizer:
    def __init__(self,
                 accent_words_dict,
                 spell_corrector,
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

    @staticmethod
    def _format(match, repl):
        return "{} {}{}".format(
            match.group(1),
            repl if match.group(2).islower() else repl.upper(),
            match.group(3))

    def raw_daw(self, match):
        """Normalize text that misuse raw and daw before noisification."""
        if match.group(1) in self.vowels:
            return self._format(match, 'r')  # raw
        elif match.group(1) in self.consonants:
            return self._format(match, 'd')  # daw

    def expand_expr(self, text):
        return self.expand_pattern.sub(self.expand_repl, text)

    def accent_styles(self, text):
        return self._substitution(text, self.accent_words_dict)

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

