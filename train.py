"""Script to do all the work:
- generate dataset
- train the model
"""
import os
import argparse
import shutil
from training.data import nosify_dataset
from utils import generate_vocab, split_dataset, sent_tokenizer
import seq2seq


def main():
    """Contains all of the work that needs to be done."""
    args = parse_args()
    model_name = args.model_name

    corpus_path = os.path.join('training', 'data', 'corpus')
    articles_path = os.path.join(corpus_path, args.src)

    training_path = os.path.join('training', 'data', 'dataset')
    os.makedirs(os.path.join(training_path, model_name), exist_ok=True)

    sent_tokenizer_path = os.path.join(
        'training', 'data', 'sent_tokenizer.pickle')

    if args.train_sent_tokenizer:
        sent_tokenizer.train(articles_path, sent_tokenizer_path)

    if args.generate_dataset:
        nosify_dataset.collect_dataset(articles_path,
                                       os.path.join(
                                           training_path, model_name,
                                           'dataset.dec'),
                                       tok=sent_tokenizer_path,
                                       char_level_emb=args.char_emb,
                                       augment_data=args.augment_data,
                                       shuffle=args.shuffle,
                                       max_seq_len=args.max_seq_len)

        split_dataset.split(
            os.path.join(
                training_path, model_name,
                'dataset.enc'),
            os.path.join(training_path, model_name),
            'enc', test_size=500, dev_size=500)

        split_dataset.split(
            os.path.join(
                training_path, model_name,
                'dataset.dec'),
            os.path.join(training_path, model_name),
            'dec', test_size=500, dev_size=500)

        generate_vocab.get_vocab(
            os.path.join(training_path, model_name, 'train.enc'),
            max_vocab_size=20000)

        generate_vocab.get_vocab(
            os.path.join(training_path, model_name, 'train.dec'),
            max_vocab_size=20000)

    if args.train:
        data_dir = os.path.join(training_path, model_name)
        model_dir = os.path.join('training', 'model', model_name)
        os.makedirs(model_dir, exist_ok=False)
        hparams_file = os.path.join('hparams.json')

        if not os.path.exists(os.path.join(model_dir, hparams_file)):
            shutil.copy(hparams_file, os.path.join(model_dir))

        hparams = seq2seq.utils.load_hparams(
            os.path.join(model_dir, hparams_file))
        trainer = seq2seq.trainer.Trainer(
            data_dir=data_dir, model_dir=model_dir, hparams=hparams)
        trainer.train()


def parse_args():
    """Parse the arguments needed before running.
    Returns:
        args : contains the source text, model name, and if
               should generate dataset or train.
    """
    parser = argparse.ArgumentParser()

    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('src', type=str, help="Filename of your source text")

    parser.add_argument('--model_name', default='model', type=str,
                        help="Unique name for each model you train.")

    parser.add_argument('--generate_dataset',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="Generate parallel noisy text")

    parser.add_argument('--train',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="Start/Resume train")

    parser.add_argument('--char_emb',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="""Embedding type
                        True for char-level, False for word-level""")

    parser.add_argument('--augment_data',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="""Augment data by adding the dataset vocabulary
                        and the n-gram from unigram to 6-gram""")

    parser.add_argument('--shuffle',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="""Shuffle the dataset""")

    parser.add_argument('--max_seq_len', default=140, type=int,
                        help="""Maximum seq length
                        to be used in dataset generation""")

    parser.add_argument('--train_sent_tokenizer',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="Train a new sentence tokenizer")

    return parser.parse_args()


if __name__ == "__main__":
    main()
