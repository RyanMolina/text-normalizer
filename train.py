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
    dataset_name = args.dataset_name
    model_name = args.model_name
    corpus_name = args.corpus_name

    FILE_PATH = os.path.dirname(__file__)
    TRAIN_PATH = os.path.join(FILE_PATH, 'training')
    MODEL_PATH = os.path.join(TRAIN_PATH, 'model', model_name)
    DATASET_PATH = os.path.join(TRAIN_PATH, 'data', 'dataset', dataset_name)
    CORPUS_PATH = os.path.join(TRAIN_PATH, 'data', 'corpus', corpus_name)


    if args.corpus_name:
        if not os.path.exists(DATASET_PATH):
            print("> Generating dataset from the corpus file...") 
            os.makedirs(DATASET_PATH, exist_ok=True)
            noisify_dataset.run(CORPUS_PATH,
                                DATASET_PATH,
                                char_level_emb=args.char_emb,
                                augment_data=args.augment_data,
                                shuffle=args.shuffle,
                                max_seq_len=args.max_seq_len)

        print("> Generating the vocab files...")
        generate_vocab.get_vocab(
            os.path.join(DATASET_PATH, 'train.enc'),
            max_vocab_size=20000)

        generate_vocab.get_vocab(
            os.path.join(DATASET_PATH, 'train.dec'),
            max_vocab_size=20000)


    if args.train:
        print("> Start trainining...")
        print(FILE_PATH)

        os.makedirs(MODEL_PATH, exist_ok=True)
        hparams_file = os.path.join(FILE_PATH, 'hparams.json')

        noisify_dataset_file = os.path.abspath(noisify_dataset.__file__)
        if not os.path.exists(os.path.join(MODEL_PATH, 'hparams.json')):
            print("> Backing up hparams.json")
            shutil.copy(hparams_file, MODEL_PATH)
        else:
            print('> Failed to back up: ', hparams_file)

        if not os.path.exists(os.path.join(DATASET_PATH, 'noisify_dataset.py')):
            print("> Backing up noisify_dataset.py")
            shutil.copy(noisify_dataset_file,
                        os.path.join(DATASET_PATH, 'noisify_dataset.py.bak'))
        else:
            print('> Failed to back up: ', noisify_dataset_file)

        hparams = utils.load_hparams(
            os.path.join(MODEL_PATH, hparams_file))

        normalizer_trainer = trainer.Trainer(
            data_dir=DATASET_PATH, model_dir=MODEL_PATH, hparams=hparams)
        normalizer_trainer.train()


def parse_args():
    """Parse the arguments needed before running.


    Returns:
        args : contains the source text, model name, and if
               should generate dataset or train.

    """
    parser = argparse.ArgumentParser()

    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--model_name', default='model', type=str,
                        help="Unique name for each model you train.")

    parser.add_argument('--dataset_name', default='dataset', type=str,
                        help="Unique name of dataset you use for training")

    parser.add_argument('--corpus_name', type=str, default=None,
                        help="Generate parallel noisy text")

    parser.add_argument('--train',
                        type="bool", nargs="?", const=True,
                        default=True,
                        help="Start/Resume train")

    parser.add_argument('--char_emb',
                        type="bool", nargs="?", const=True,
                        default=True,
                        help="""Embedding type
                        True for char-level, False for word-level""")

    parser.add_argument('--augment_data',
                        type="bool", nargs="?", const=True,
                        default=True,
                        help="""Augment data by adding the dataset vocabulary
                        and the n-gram from unigram to 6-gram""")

    parser.add_argument('--shuffle',
                        type="bool", nargs="?", const=True,
                        default=True,
                        help="""Shuffle the dataset""")

    parser.add_argument('--max_seq_len', default=140, type=int,
                        help="""Maximum seq length
                        to be used in dataset generation""")

    return parser.parse_args()


if __name__ == "__main__":
    main()
