import re
from collections import Counter

class SpellCorrector:
    def __init__(self, dict_path):
        self.WORDS = Counter(
            re.findall(r'\w+', open(dict_path, 'r').read()))

        self.vowels = 'aeiou'
        self.vowels += self.vowels.upper()
        self.consonants = "bcdfghjklmnpqrstvwxyz"
        self.consonants += self.consonants.upper()
        self.alphabet = self.vowels + self.consonants

    def P(self, word):
        N = sum(self.WORDS.values())
        return self.WORDS[word] / N

    def _one_char_edits(self, word):
        """Based on Peter Norvig's Spell Corrector."""
        splits = [(word[:i], word[i:])
                  for i in range(len(word) + 1)]

        deletes = [l + r[1:]
                   for l, r in splits
                   if r]

        transposes = [l + r[1] + r[0] + r[2:]
                      for l, r in splits
                      if len(r) > 1]

        replaces = [l + c + r[1:]
                    for l, r in splits
                    for c in self.alphabet]
                        
        inserts = [l + c + r
                   for l, r in splits
                   for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)
    
    def _two_char_edits(self, word):
        return (e2
                for e1 in self._one_char_edits(word)
                for e2 in self._one_char_edits(e1))

    def correction(self, word):
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        return (self.known([word])
                or self.known(self._one_char_edits(word))
                or self.known(self._two_char_edits(word))
                or [word])

    def known(self, words):
        return set(word for word in words if word in self.WORDS)