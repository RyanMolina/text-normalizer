"""
Generate vocabulary for a tokenized text file.
"""

import sys
import argparse
import collections
import logging
import os


def main():
    get_vocab(args.infile,
              args.max_vocab_size,
              args.delimiter,
              args.downcase,
              args.min_frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate vocabulary for a tokenized text file.")
    parser.add_argument(
        "--min_frequency",
        dest="min_frequency",
        type=int,
        default=0,
        help="Minimum frequency of a word to be included in the vocabulary.")
    parser.add_argument(
        "--max_vocab_size",
        dest="max_vocab_size",
        type=int,
        help="Maximum number of tokens in the vocabulary")
    parser.add_argument(
        "--downcase",
        dest="downcase",
        type=bool,
        help="If set to true, downcase all text before processing.",
        default=False)
    parser.add_argument(
        "infile",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Input tokenized text file to be processed.")
    parser.add_argument(
        "--delimiter",
        dest="delimiter",
        type=str,
        default=" ",
        help="""Delimiter character for tokenizing. Use \" \" and \"\" for word
            and char level respectively."""
    )
    args = parser.parse_args()
    main()


def get_vocab(infile, max_vocab_size=None, delimiter=" ",
              downcase=False, min_frequency=0, to_file=True):
    # Counter for all tokens in the vocabulary
    cnt = collections.Counter()
    with open(infile, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if downcase:
                line = line.lower()
            if delimiter == "":
                tokens = list(line.strip())
            else:
                # tokens = nltk.word_tokenize(line.strip())
                tokens = line.strip().split(delimiter)
            tokens = [_ for _ in tokens if len(_) > 0]
            cnt.update(tokens)

    # Filter tokens below the frequency threshold
    if min_frequency > 0:
        filtered_tokens = [(w, c) for w, c in cnt.most_common()
                           if c > min_frequency]
        cnt = collections.Counter(dict(filtered_tokens))

    logging.info("Found %d unique tokens with frequency > %d.",
                 len(cnt), min_frequency)

    # Sort tokens by 1. frequency 2. lexically to break ties
    word_with_counts = cnt.most_common()
    word_with_counts = sorted(
        word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

    # Take only max-vocab
    if max_vocab_size is not None:
        word_with_counts = word_with_counts[:max_vocab_size]

    path, filename = os.path.split(infile)
    filename, extension = os.path.splitext(filename)

    if to_file:
        with open(os.path.join(path, "vocab" + extension), 'w') as outfile:
            print("<unk>", file=outfile)
            print("<s>", file=outfile)
            print("</s>", file=outfile)
            for word, count in word_with_counts:
                print("{}".format(word), file=outfile)

    return word_with_counts
