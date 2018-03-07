"""Contains the Serve class."""
import os
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from .training.data.textnormalizer import TextNormalizer
from .training.data.spellcorrector import SpellCorrector
from .utils.helper import csv_to_dict

class Serve:
    """Serve an instance of the trained model."""

    def __init__(self, sess, model_name, dataset_name, checkpoint, char_emb=False, mitigate_problems=False):
        """Prepare the model's dataset and trained model."""
        os.makedirs(os.path.join('training', 'data', 'dataset', dataset_name),
                    exist_ok=True)
        CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        TRAINING_PATH = os.path.join(CURRENT_PATH, 'training') 

        data_dir = os.path.join(TRAINING_PATH, 'data', 'dataset', dataset_name)
        model_dir = os.path.join(TRAINING_PATH, 'model', model_name)

        hparams = utils.load_hparams(
            os.path.join(model_dir, 'hparams.json'))

        self.detokenizer = MosesDetokenizer()
        
        self.char_emb = char_emb
        self.normalizer = predictor.Predictor(sess,
                                              dataset_dir=data_dir,
                                              output_dir=model_dir,
                                              output_file=checkpoint,
                                              hparams=hparams)
        self.mitigate_problems = mitigate_problems
        if self.mitigate_problems: 
            ACCENT_PATH = os.path.join(TRAINING_PATH, 'data', 'accented_words.dic')
            PANDIWA_PATH = os.path.join(TRAINING_PATH, 'data', 'pandiwa.dic')
        
            accent_words_dict = csv_to_dict(ACCENT_PATH)
            accent_words_dict = {v2: k
                                for k, v in accent_words_dict.items()
                                for v2 in v}

            pprint(accent_words_dict)
            with open(PANDIWA_PATH, 'r') as pandiwa_file:
                pandiwa_words_dict = pandiwa_file.read().splitlines()

            with open(os.path.join(TRAINING_PATH, 'data', 'hyph_fil.tex'),
                        'r') as f:
                hyphenator_dict = f.read()

            spell_corrector = SpellCorrector(
                    dict_path=os.path.join(
                        TRAINING_PATH, 'data', 'corpus', 'tagalog_sent_v3.txt'))

            self.t_normalizer = TextNormalizer(
                    accent_words_dict=accent_words_dict,
                    hyphenator_dict=hyphenator_dict,
                    pandiwa_words_dict=pandiwa_words_dict,
                    spell_corrector=spell_corrector)


    def model_api(self, input_data):
        """Input preprocessing before prediction.

        Args:
            input_data (string): informal text
        Returns:
            string: normalized text
        """
        # TODO: Detect if word is in normal form to avoid FP results.
        # TODO: Allow punctuations on input.
        # TODO: Create a model to detect named-entity
        # TODO: Detokenize here as last step.

        output = ""
        for sentence in sent_tokenize(input_data):
            if str.isupper(sentence):
                sentence = sentence.lower()

            if self.char_emb:
                tokens = ' '.join(word_tokenize(sentence)) \
                    .replace('``', '"') \
                    .replace("''", '"')

                tokens = self._char_emb_format(tokens)
            else:
                tokens = sentence

            # tokens = tokens.lower()
            normalized = self.normalizer.predict(tokens)

            if self.char_emb:
                normalized = normalized.replace(' ', '') \
                                       .replace('<space>', ' ')

                normalized = normalized.replace('<lquotes>', '``') \
                                       .replace('<rquotes', "''")

                normalized = normalized.split()
                
                if self.mitigate_problems:
                    normalized = ' '.join([self.t_normalizer.spell_correct(word)
                                for word in normalized])

                    normalized = self.t_normalizer.expand_expr(normalized)

                    normalized = self.t_normalizer.raw_daw.sub(
                            self.t_normalizer.raw_daw_repl, normalized)

                    normalized = normalized.split()

                    normalized = [self.t_normalizer.accent_style(word)
                                for word in self.t_normalizer.mwe_tokenizer \
                                                .tokenize(normalized)]

                normalized = self.detokenizer.detokenize(normalized,
                                                         return_str=True)
                

            output += normalized + " "
        return output.strip()

    @staticmethod
    def _char_emb_format(text):
        return ' '.join(list(text)).replace(' ' * 3, ' <space> ')


def parse_args():
    """Parse the arguments needed before running.

    Returns:
        args: contains the parsed arguments

    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Dir of your selected model and the checkpoint.")

    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--model_name', type=str,
                        help="""
                        Name of the model to use.
                        Change only if you want to try other models.
                        Models must be saved in training/model/<model_name>
                        """)

    parser.add_argument('--dataset_name', type=str,
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
    parser.add_argument('--char_emb',
                        type="bool", nargs="?", const=False,
                        default=False,
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
    parser.add_argument('--gpu_mode',
                        type="bool", nargs="?", const=False,
                        default=False,
                        help="""Use GPU in computation instead of CPU.""")
    return parser.parse_args()


if __name__ == '__main__':
    #  Imports when running as a script
    import tensorflow as tf
    from seq2seq import predictor, utils

    ARGS = parse_args()

    if not ARGS.gpu_mode:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # TODO: Break-word <= character limit

    with tf.Session() as sess:
        NORMALIZER = Serve(sess=sess,
                           model_name=ARGS.model_name,
                           dataset_name=ARGS.dataset_name,
                           checkpoint=ARGS.checkpoint,
                           char_emb=ARGS.char_emb)

        #  Input mode using file and generate results.txt file
        if ARGS.input_file:
            normalized = []
            path, _ = os.path.split(ARGS.input_file)
            with open(ARGS.input_file, 'r') as infile,\
                    open(os.path.join(path, 'results.txt'), 'w') as outfile:
                contents = infile.read()

                enc_rows = contents.splitlines()

                for row in enc_rows:
                    result = NORMALIZER.model_api(row)
                    normalized.append(result)
                    outfile.write(result + "\n")

            #  Check the accuracy
            if ARGS.expected_file:
                correct = 0
                with open(ARGS.expected_file, 'r') as expected_file:
                    contents = expected_file.read()
                    dec_rows = contents.splitlines()

                    for i, e in enumerate(normalized):
                        if not ARGS.char_emb:
                            print('-' * 30)
                            print('input:  ' + enc_rows[i])
                            print("system: " + e)
                            print("expect: " + dec_rows[i])
                        else:
                            print('-' * 30)
                            print('input:  ' + enc_rows[i]
                                  .replace(' ', '').replace('<space>', ' '))
                            print("system: " + e.replace(' ', '')
                                  .replace('<space>', ' '))
                            print("expect: " + dec_rows[i]
                                  .replace(' ', '').replace('<space>', ' '))

                        if e.lower()[:280] == dec_rows[i].lower()[:280]:
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
