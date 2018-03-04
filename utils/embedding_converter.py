"""Module to convert dataset embedding type."""
import argparse


def char_embedding(text):
    """Turn dataset to space seperated characters and encode ' ' to <space>."""
    text = ' '.join(list(text)).replace(' '*3, ' <space> ')
    text = text.replace('` `', '<lquotes>') \
                    .replace("' '", '<rquotes>')
    return text

def word_embedding(text):
    """From char_embedding to word_embedding."""
    return text.replace(' ', '').replace('<space>', ' ')


def parse_args():
    """Command-line ArgumentParser."""
    parser = argparse.ArgumentParser(
        description="python3 embedding_converter.py <file_name> <mode>")

    parser.add_argument('src', type=str, help="File to be converted")
    parser.add_argument('embedding', type=str,
                        help='Select the type of embedding "char" or "word"')

    return parser.parse_args()


def main():
    """Convert the specified file to the selected embedding."""
    args = parse_args()
    with open(args.src, 'r') as infile:
        contents = infile.read()

    if args.embedding == "char":
        contents = char_embedding(contents)
    elif args.embedding == "word":
        contents = word_embedding(contents)
    else:
        raise ValueError("{} is not an embedding type".format(args.embedding))

    with open(args.src, 'w') as outfile:
        for row in contents.splitlines():
            outfile.write(row + "\n")


if __name__ == '__main__':
    main()