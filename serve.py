"""Contains the Serve class"""
import os
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize
from .seq2seq import predictor, utils


class Serve:
    """Serve an instance of the trained model"""
    def __init__(self, sess, model_name, checkpoint, char_emb=False):
        os.makedirs(os.path.join('training', 'data', 'dataset', model_name),
                    exist_ok=True)
        cwd = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(cwd, 'training', 'data', 'dataset', model_name)
        model_dir = os.path.join(cwd, 'training', 'model', model_name)

        hparams = utils.load_hparams(
            os.path.join(model_dir, 'hparams.json'))

        self.char_emb = char_emb
        self.normalizer = predictor.Predictor(sess,
                                              dataset_dir=data_dir,
                                              output_dir=model_dir,
                                              output_file=checkpoint,
                                              hparams=hparams)

    def model_api(self, input_data):
        """Does all the preprocessing before prediction.
        1. Split into sentences
        2. Tokenize to words
        3. Tokenize into characters
        4. encode ' ' into <space>
        Args:
            input_data (string): informal text
        Returns:
            string: normalized text
        """

        output = ""
        for sentence in sent_tokenize(input_data):
            if not self.char_emb:
                tokens = ' '.join(word_tokenize(sentence))
            else:
                tokens = self._char_emb_format(sentence)

            normalized = self.normalizer.predict(tokens)

            if self.char_emb:
                normalized = normalized.replace(' ', '').replace('<space>', ' ')

            output += normalized
        return output

    @staticmethod
    def _char_emb_format(text):
        return ' '.join(list(text)).replace(' ' * 3, ' <space> ')


def parse_args():
    """Parse the arguments needed before running.
    Returns:
        args: contains the parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="Dir of your selected model and the checkpoint.")

    parser.add_argument('--model_name', default='model_served', type=str,
                        help="""
                        Name of the model to use.
                        Change only if you want to try other models.
                        (Default: 'model_served')
                        """)
    parser.add_argument('--checkpoint', default=None, type=str,
                        help="""
                        Specify the checkpoint filename.
                        (Default: latest checkpoint)
                        """)
    parser.add_argument('--char_emb', default=False, type=bool,
                        help="""
                        Char-level or word-level embedding
                        """
                        )
    return parser.parse_args()


if __name__ == '__main__':
    import tensorflow as tf
    ARGS = parse_args()
    with tf.Session() as sess:
        NORMALIZER = Serve(sess=sess,
                           model_name=ARGS.model_name,
                           checkpoint=ARGS.checkpoint,
                           char_emb=ARGS.char_emb)
        while True:
            text = input("> ")
            if text:
                print(NORMALIZER.model_api(text))
            else:
                break
