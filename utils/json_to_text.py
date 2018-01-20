import json
import argparse


def main():
    articles = json.load(open(args.src))
    with open(args.out, 'w') as f:
        for article in articles:
            print(article['title'].strip(), file=f)
            print(article['body'].strip(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help="Filename to json file")
    parser.add_argument('out', type=str, help="Filename to output text file")
    args = parser.parse_args()
    main()
