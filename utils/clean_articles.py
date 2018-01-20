import re
import argparse

spaces = re.compile(r'\s+')
ascii_only = re.compile(r'[^\x00-\x7f]')


def main():
    with open(args.src, 'r') as in_file, \
         open(args.out, 'w') as out_file:
        articles = in_file.read()
        articles = ascii_only.sub('', articles).splitlines()
        for article in articles:
            article = spaces.sub(' ', article)
            article = article.replace('\r', '').replace('\n', '')
            if article:
                print(article.strip(), file=out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help="Filename to text file")
    parser.add_argument('out', type=str, help="Filename to output text file")
    args = parser.parse_args()
    main()
