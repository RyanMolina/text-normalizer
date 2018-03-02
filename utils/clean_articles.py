import re
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize

spaces = re.compile(r'\s+') # Multiple spaces
ascii_only = re.compile(r'[^\x00-\x7f]') # Removing the non-ascii characters
news_location = re.compile(r'^(\w+\s\w+|\w+),\s\w+\s-\s') # MANILA, Philippines - <Article>

common_informal_words = {'meron': 'mayroon',
                         'penge': 'pahingi',
                         'kundi': 'kung hindi',
                         'tsaka': 'saka'}

vowels = 'aeiou'
vowels += vowels.upper()
consonants = "bcdfghjklmnpqrstvwxyz"
consonants += consonants.upper()
alphabet = vowels + consonants

raw_daw = \
    re.compile(r'\b([^aeiou]|[aeiou])\b\s([dr])(aw|ito|oon|in)',
                re.IGNORECASE)

def main():
    with open(args.src, 'r') as in_file, \
         open(args.out, 'w') as out_file:
        articles = in_file.read()
        articles = ascii_only.sub('', articles).splitlines()
        for article in articles:
            article = spaces.sub(' ', article)
            article = news_location.sub('', article)
            article = article.replace('\r', '').replace('\n', '')
            if article:
                sentence = article
                # for sentence in sent_tokenize(article): 
                tokens = word_tokenize(sentence)
                sentence = ' '.join(tokens)
                print(sentence.strip(), file=out_file)

def _format(match, repl):
    return "{} {}{}".format(
        match.group(1),
        repl if match.group(2).islower() else repl.upper(),
        match.group(3))

def normalize_raw_daw(match):
    """Normalize text that misuse raw and daw before noisification."""
    if match.group(1) in vowels:
        return _format(match, 'r')  # raw
    elif match.group(1) in consonants:
        return _format(match, 'd')  # daw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help="Filename to text file")
    parser.add_argument('out', type=str, help="Filename to output text file")
    args = parser.parse_args()
    main()
