"""Script to do all the work."""
import os
import argparse
import shutil
try:
    from .seq2seq import utils, trainer
    from .training.data import noisify_dataset
    from .utils import generate_vocab, split_dataset
except SystemError:
    from training.data import noisify_dataset
    from utils import generate_vocab, split_dataset
    from seq2seq import utils, trainer


def main():
    """Contains all of the work that needs to be done."""
    args = parse_args()
    model_name = args.model_name

    FILE_PATH = os.path.dirname(__file__)
    training_path = os.path.join(FILE_PATH, 'training', 'data', 'dataset')
    os.makedirs(os.path.join(training_path, model_name), exist_ok=True)

    if args.corpus:
        print("> Generating dataset from the corpus file...")
        corpus_path = os.path.join(FILE_PATH, 'training', 'data', 'corpus')
        articles_path = os.path.join(corpus_path, args.corpus)
        noisify_dataset.collect_dataset(articles_path,
                                        os.path.join(training_path,
                                                     model_name,
                                                     'dataset.dec'),
                                        char_level_emb=args.char_emb,
                                        augment_data=args.augment_data,
                                        shuffle=args.shuffle,
                                        max_seq_len=args.max_seq_len)

    if args.split_dataset:
        print("> Splitting the dataset to train/dev/test set...")
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

    if args.generate_vocab:
        print("> Generating the vocab files...")
        generate_vocab.get_vocab(
            os.path.join(training_path, model_name, 'train.enc'),
            max_vocab_size=20000)

        generate_vocab.get_vocab(
            os.path.join(training_path, model_name, 'train.dec'),
            max_vocab_size=20000)

    if args.train:
        print("> Start trainining...")
        print(FILE_PATH)
        data_dir = os.path.join(FILE_PATH, training_path, model_name)
        model_dir = os.path.join(FILE_PATH, 'training', 'model', model_name)
        os.makedirs(model_dir, exist_ok=True)
        hparams_file = os.path.join(FILE_PATH, 'hparams.json')
        noisify_dataset_file = os.path.abspath(noisify_dataset.__file__)

        if not os.path.exists(os.path.join(model_dir, hparams_file)):
            print("> Backing up hparams.json")
            shutil.copy(hparams_file, os.path.join(model_dir))

        if not os.path.exists(os.path.join(data_dir, 'noisify_dataset.py')):
            print("> Backing up noisify_dataset.py")
            shutil.copy(noisify_dataset_file,
                        os.path.join(data_dir, 'noisify_dataset.py.bak'))

        hparams = utils.load_hparams(
            os.path.join(model_dir, hparams_file))
        normalizer_trainer = trainer.Trainer(
            data_dir=data_dir, model_dir=model_dir, hparams=hparams)
        normalizer_trainer.train()


def parse_args():
    """Parse the arguments needed before running.

    Returns:
        args : contains the source text, model name, and if
               should generate dataset or train.

    """
    parser = argparse.ArgumentParser()

    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('model_name', default='model', type=str,
                        help="Unique name for each model you train.")

    parser.add_argument('--corpus', type=str, default=None,
                        help="Generate parallel noisy text")

    parser.add_argument('--split_dataset',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="""Split dataset to train/dev/test""")

    parser.add_argument('--generate_vocab',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="""Generate the vocab file from train file""")

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

    return parser.parse_args()


if __name__ == "__main__":
    main()
