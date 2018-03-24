"""Contains the Serve class."""
import os
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize
# from utils.tokenizer import word_tokenize, sent_tokenize


class Serve:
    """Serve an instance of the trained model."""

    def __init__(self, sess, model_name, checkpoint, char_emb=False, max_seq_len=140):
        """Prepare the model's dataset and trained model."""
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
        """Input preprocessing before prediction.

        Args:
            input_data (string): informal text
        Returns:
            string: normalized text
        """
        output = ""
        for sentence in sent_tokenize(input_data):
            sentence = sentence[:max_seq_len]
            max_seq_len = sentence.rfind(' ')
            sentence = sentence[:max_seq_len]
            tokens = self._char_emb_format(sentence)

            normalized = self.normalizer.predict(tokens)

            if self.char_emb:
                normalized = normalized.replace(' ', '') \
                                       .replace('<space>', ' ')

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

    parser.add_argument('model_name', type=str,
                        help="""
                        Name of the model to use.
                        Change only if you want to try other models.
                        Models must be saved in training/model/<model_name>
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
    parser.add_argument('--input_file', type=str, default=None,
                        help="""
                        File you want to normalize
                        """
                        )
    parser.add_argument('--expected_file', type=str, default=None,
                        help="""
                        If you want to see the accuracy.
                        """
                        )

    return parser.parse_args()


if __name__ == '__main__':
    #  Imports when running as a script
    import tensorflow as tf
    from seq2seq import predictor, utils

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ARGS = parse_args()
    with tf.Session() as sess:
        NORMALIZER = Serve(sess=sess,
                           model_name=ARGS.model_name,
                           checkpoint=ARGS.checkpoint,
                           char_emb=ARGS.char_emb)
        #  Input mode using file and generate results.txt file
        if ARGS.input_file:
            normalized = []
            path, _ = os.path.split(ARGS.input_file)
            print(path)
            with open(ARGS.input_file, 'r') as infile,\
                    open(os.path.join(path, 'results.txt'), 'w') as outfile:
                contents = infile.read()

                rows = contents.splitlines()

                for row in rows:
                    result = NORMALIZER.model_api(row)
                    normalized.append(result)
                    outfile.write(result + "\n")

            #  Check the accuracy
            if ARGS.expected_file:
                correct = 0
                with open(ARGS.expected_file, 'r') as expected_file:
                    contents = expected_file.read()
                    rows = contents.splitlines()

                    for i, e in enumerate(normalized):
                        print("system: " + e)
                        print("expect: " + rows[i])
                        if e == rows[i]:
                            correct += 1
                    print("Correctly normalized: {} Total: {} Accuracy: {}"
                          .format(correct,
                                  len(normalized),
                                  correct/len(normalized) * 100))

        #  Input mode using console's input and print the normalized text
        else:
            while True:
                text = input("> ")
                if text:
                    print(NORMALIZER.model_api(text))
                else:
                    break

else:
    from .seq2seq import predictor, utils
