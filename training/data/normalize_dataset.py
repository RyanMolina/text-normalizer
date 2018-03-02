import os
import re
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize
from .textnormalizer import TextNormalizer

spaces = re.compile(r'\s+') # Multiple spaces
ascii_only = re.compile(r'[^\x00-\x7f]') # Removing the non-ascii characters
news_location = re.compile(r'^(\w+\s\w+|\w+),\s\w+\s-\s') # MANILA, Philippines - <Article>

dict_path = os.path.join('..', 'training', 'data', 'corpus', 'vocab.txt')
accent_path = os.path.join('..', 'training', 'data', 'accented_words.dic')
hyph_path = os.path.join('..', 'training', 'data', 'hyph_fil.tex')

def main():
    text_normalizer = TextNormalizer(accent_words_dict=)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help="Filename to text file")
    parser.add_argument('out', type=str, help="Filename to output text file")
    args = parser.parse_args()
    main()
