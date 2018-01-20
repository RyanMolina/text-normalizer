import argparse
import pickle
from nltk.tokenize.punkt import (PunktSentenceTokenizer,
                                 PunktLanguageVars,
                                 PunktTrainer)


def train(src, tgt):
    with open(src, 'r', encoding='utf-8') as infile, \
            open(tgt, 'wb') as sent_tokenizer:
        contents = infile.read()
        language_punkt_vars = PunktLanguageVars
        # language_punkt_vars.sent_end_chars=tuple(args.end_chars)
        print("# Training sent tokenizer")
        trainer = PunktTrainer(contents, language_punkt_vars)
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.INCLUDE_ABBREV_COLLOCS = True
        params = trainer.get_params()
        tokenizer = PunktSentenceTokenizer(params)
        tokenizer._params.abbrev_types.add('brgy')
        tokenizer._params.abbrev_types.add('sen')
        tokenizer._params.abbrev_types.add('supt')
        tokenizer._params.abbrev_types.add('rep')
        tokenizer._params.abbrev_types.add('dr')
        tokenizer._params.abbrev_types.add('col')
        tokenizer._params.abbrev_types.add('sec')
        tokenizer._params.abbrev_types.add('mt')
        tokenizer._params.abbrev_types.add('asst')
        tokenizer._params.abbrev_types.add('mr')
        tokenizer._params.abbrev_types.add('c/insp')
        tokenizer._params.abbrev_types.add('rep')
        tokenizer._params.abbrev_types.add('sta')
        tokenizer._params.abbrev_types.add('sto')
        pickle.dump(tokenizer, sent_tokenizer)


def main():
    args = parse_args()
    train(args.src, args.out)


def parse_args():
    parser = argparse.ArgumentParser(description="Opens a file and learn \
            boundary detection")
    parser.add_argument('src', type=str,
                        help="Filename of the train data")
    parser.add_argument('--out', type=str,
                        default="trained_sent_tokenizer.pickle",
                        help="Filename of the trained PunktSentenceTokenizer")
    # parser.add_argument('--end_chars', nargs='+', default=['!', '.', '?'],
    #                     help="List of end characters")
    return parser.parse_args()


if __name__ == '__main__':
    main()

"""
It uses an unsupervised learning algorithm. The specific technique
used in this case is called sentence boundary detection and it works
by counting punctuation and tokens that commonly end a sentence,
such as a period or new line, then using the resulting frequencies
to decide what the sentence boundaries should actually look like.
"""
