"""
This module contains all the utility function
needed in the model.
"""
import json
import codecs
import tensorflow as tf
import numpy as np
import _locale
import sys
import math
import collections
import time

_locale.getdefaultlocale = (lambda *args: 'utf-8')


def check_tensorflow_version():
    """Checks Tensorflow version to make sure everything works.

    Raises:
        EnvironmentError: [description]
    """

    min_tf_version = "1.4.0"
    if tf.__version__ < min_tf_version:
        raise EnvironmentError("TensorFlow version must be >= {}"
                               .format(min_tf_version))


def load_hparams(hparams_file):
    """Load the json file into python

    Args:
        hparams_file (str): path to the hyperparameter file

    Raises:
        FileNotFoundError: raise file not found error

    Returns:
        HParams: HParams instances loaded with hparams_file values.
    """

    if tf.gfile.Exists(hparams_file):
        print("+ Loading hparams from {} ...".format(hparams_file))
        with codecs.getreader('utf-8')(tf.gfile.GFile(
                                       hparams_file, 'rb')) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
                return hparams
            except ValueError:
                print('- Error loading hparams file.')
    else:
        raise FileNotFoundError(hparams_file + " not found.")


def safe_exp(value):
    """Exponentiation with safety catch.

    Args:
        value (float): the value to be exponentiated

    Returns:
        float: exponentiated or inf value
    """

    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def print_time(s, start_time):
    """Take a start time, print elapsed duration,
    and return a new time.

    Args:
        s (string): display message
        start_time (float): start time
    """

    print("{}, time {}s, {}."
          .format(s, (time.time() - start_time),
                  time.ctime()))
    sys.stdout.flush()


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush
    and print simultaneously to a file.
    Args:
        s (string): string to be printed
        f (file, optional): Defaults to None.
                            Print to file
        new_line (bool, optional): Defaults to True.
                                   If you want to print new line
    """

    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def print_hparams(hparams):
    """Print hparams, can skip keys based on patterns.

    Args:
        hparams (HParams): hold a set of hparams as name-value pairs
    """

    values = hparams.values()
    for key in sorted(values.keys()):
        print_out("  {}={}".format(key, str(values[key])))


def format_text(words):
    """Combines the byte-string and decode it to utf-8

    Args:
        words (list): list of byte-string

    Returns:
        string: decoded to utf-8
    """
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words).decode('utf-8')


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the train graph.

    Args:
        summary_writer (tf.summary.FileWriter): FileWriter object
        global_step (int): The current train step.
        tag (string): Tag or name of value.
        value (float): Value to be added on summary.
    """

    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def unk_replace(source_tokens, predicted_tokens, attention_scores):
    """Replace the <unk> tokens with the aligned values
    from the source tokens using attention scores.

    Args:
        source_tokens (list): List of byte-strings
        predicted_tokens (list): List of byte-strings
        attention_scores (numpy): Numpy array attention scores

    Returns:
        list: List of byte-string
    """

    result = []
    for token, scores in zip(predicted_tokens, attention_scores):
        if token == b"<unk>":
            max_score_index = np.argmax(scores)
            chosen_source_token = source_tokens[max_score_index]
            new_target = chosen_source_token
            result.append(new_target)
        else:
            result.append(token)
    return result
