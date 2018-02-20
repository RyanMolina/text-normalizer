"""A list of (regexp, repl) pairs applied in sequence.

The resulting string is split on whitespace.
(Adapted from the Punkt Word Tokenizer)
"""
import re

_tokenize_regexps = [
    # uniform quotes
    (re.compile(r'\'\''), r'"'),
    (re.compile(r'``'), r'"'),

    # Separate punctuation (except period) from words:
    (re.compile(r'(^|\s)(\')'), r'\1\2 '),
    (re.compile(r'(?=[(\"`{\[:;&#*@])(.)'), r'\1 '),
    (re.compile(r'(.)(?=[?!)\";}\]*:@\'])'), r'\1 '),
    (re.compile(r'(?=[)}\]])(.)'), r'\1 '),
    (re.compile(r'(.)(?=[({\[])'), r'\1 '),
    (re.compile(r'((^|\s)-)(?=[^\-])'), r'\1 '),

    # Treat double-hyphen as one token:
    (re.compile(r'([^-])(--+)([^-])'), r'\1 \2 \3'),
    (re.compile(r'(\s|^)(,)(?=(\S))'), r'\1\2 '),

    # Only separate comma if space follows:
    (re.compile(r'(.)(,)(\s|$)'), r'\1 \2\3'),

    # Combine dots separated by whitespace to be a single token:
    (re.compile(r'\.\s\.\s\.'), r'...'),

    # Separate words from ellipses
    (re.compile(r'([^.]|^)(\.{2,})(.?)'), r'\1 \2 \3'),
    (re.compile(r'(^|\s)(\.{2,})([^.\s])'), r'\1\2 \3'),
    (re.compile(r'([^.\s])(\.{2,})($|\s)'), r'\1 \2\3'),

    # fix %, $, &
    (re.compile(r'(\d)%'), r'\1 %'),
    (re.compile(r'\$(\.?\d)'), r'$ \1'),
    (re.compile(r'(\w)& (\w)'), r'\1&\2'),
    (re.compile(r'(\w\w+)&(\w\w+)'), r'\1 & \2'),

    # words with '
    (re.compile(r"(')(\s)([A-Za-z]+)"), r'\1\3'),
    (re.compile(r"\s('[A-Za-z])"), r"\1"),

    # quotes enclosed
    (re.compile(r"(['\"])(\w+)(['\"])"), r"`` \2 ''"),

    (re.compile(r'\s+'), r' '),

    # Split periods
    (re.compile(r'(\w+)(\.)'), r'\1 \2')]


def mwe_tokenize(s, mwes=None, sep=' '):
    """Tokenize a string into list of multiword expression."""
    if not mwes:
        mwes = []

    for mwe in mwes:
        mwe = sep.join(list(mwe))
        if mwe in s:
            s = s.replace(mwe, mwe.replace(' ', '<mwe>'))

    tokens = word_tokenize(s)
    for i, e in enumerate(tokens):
        if '<mwe>' in e:
            tokens[i] = e.replace('<mwe>', ' ')
    return tokens


def word_tokenize(s):
    """Tokenize a string using the rule above into list of words."""
    for (regexp, repl) in _tokenize_regexps:
        s = regexp.sub(repl, s)
    return s.split()


def sent_tokenize(s):
    """Tokenize a string into list of sentences."""
    return re.split(r"(?<!\w\.\w.)"
                    r"(?<![A-Z][a-z]\.)"
                    r"(?<![A-Z][a-z][a-z]\.)"
                    r"(?<![A-Z][a-z][a-z][a-z]\.)"
                    r"(?<=[.?])\s", s)


if __name__ == '__main__':
    # with open('training/data/corpus/tagalog.txt', 'r') as infile:
    #     rows = infile.read().splitlines()
    #     outfile = open('training/data/corpus/tagalog_sent_v2.txt', 'w')
    #     for row in rows:
    #         sentences = sent_tokenize(row)
    #         for sentence in sentences:
    #             outfile.write(sentence + "\n")
    #     outfile.close()
    print(word_tokenize("kalabit-pahingi kalibit/pahingi"))
