"""Contains Dataset class, BatchedInput class
and module function check_vocab, create_vocab_tables
"""
import codecs
import os
import collections
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from seq2seq import utils


class Dataset:
    """Dataset class contains the hyperparameters, dataset, and vocabulary"""

    def __init__(self, dataset_dir, hparams=None, training=True):

        if hparams is None:
            self.hparams = utils.load_hparams(dataset_dir)
        else:
            self.hparams = hparams

        self.src_max_len = self.hparams.src_max_len
        self.tgt_max_len = self.hparams.tgt_max_len
        self.dataset = None
        self.dataset_ids = None

        src_vocab_file = os.path.join(dataset_dir, 'vocab.enc')
        tgt_vocab_file = os.path.join(dataset_dir, 'vocab.dec')

        self.src_vocab_size, _ = check_vocab(src_vocab_file)
        self.tgt_vocab_size, _ = check_vocab(tgt_vocab_file)

        self.src_vocab_table, self.tgt_vocab_table = create_vocab_tables(
            src_vocab_file,
            tgt_vocab_file,
            self.hparams.unk_id)
        if training:
            self.reverse_vocab_table = None
        else:
            self.reverse_vocab_table = \
                lookup_ops.index_to_string_table_from_file(
                    tgt_vocab_file, default_value=self.hparams.unk_token)
            with open(os.path.join(dataset_dir, 'test.enc'), 'r') as enc, \
                    open(os.path.join(dataset_dir, 'test.dec'), 'r') as dec:
                self.sample_src_data = enc.read().split('\n')
                self.sample_tgt_data = dec.read().split('\n')

        self._load_dataset(dataset_dir)
        self._convert_to_tokens()

    def get_training_batch(self, num_threads=4):
        """Returns the training batch
            num_threads (int, optional): Defaults to 4.
                        parallelizing mapping of dataset.
        Returns:
            BatchedInput: contains the batch iterator, target and source.
        """

        buffer_size = self.hparams.batch_size * 400
        train_set = self.dataset_ids.shuffle(buffer_size=buffer_size)

        train_set = train_set.map(
            lambda src, tgt:
            (src, tf.concat(([self.hparams.sos_id],
                            tgt), 0),
             tf.concat((tgt, [self.hparams.eos_id]), 0)),
            num_parallel_calls=num_threads).prefetch(buffer_size)

        train_set = train_set.map(
            lambda src, tgt_in, tgt_out:
            (src, tgt_in, tgt_out,
             tf.size(src), tf.size(tgt_in)),
            num_parallel_calls=num_threads).prefetch(buffer_size)

        def batching_func(data):
            """Pad the source sequences with eos tokens."""

            return data.padded_batch(
                self.hparams.batch_size,
                padded_shapes=(tf.TensorShape([None]),
                               tf.TensorShape([None]),
                               tf.TensorShape([None]),
                               tf.TensorShape([]),
                               tf.TensorShape([])),
                padding_values=(self.hparams.eos_id,
                                self.hparams.eos_id,
                                self.hparams.eos_id,
                                0,
                                0)
            )

        if self.hparams.num_buckets > 1:
            bucket_width = (self.src_max_len + self.hparams.num_buckets - 1) \
                // self.hparams.num_buckets

            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                """Maps every element in the dataset to a key."""

                bucket_id = tf.maximum(
                    src_len // bucket_width, tgt_len // bucket_width)

                return tf.to_int64(tf.minimum(
                    self.hparams.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                """Receives a block of window_size elements.
                Then, call the batching_func to pad the data.
                """

                return batching_func(windowed_data)

            batched_dataset = train_set.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func,
                    reduce_func=reduce_func,
                    window_size=self.hparams.batch_size))
        else:
            batched_dataset = batching_func(train_set)

        batched_iter = batched_dataset.make_initializable_iterator()

        (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = \
            (batched_iter.get_next())

        return BatchedInput(initializer=batched_iter.initializer,
                            source=src_ids,
                            target_input=tgt_input_ids,
                            target_output=tgt_output_ids,
                            source_sequence_length=src_seq_len,
                            target_sequence_length=tgt_seq_len)

    def get_inference_batch(self, src_dataset, batch_size):
        infer_dataset = src_dataset.map(
            lambda src: tf.string_split([src]).values)

        # Splice data up to max_len only
        infer_dataset = infer_dataset.map(
            lambda src: src[:self.hparams.src_max_len_infer])

        # Convert data to ids. (id = position in lookup table)
        infer_dataset_ids = infer_dataset.map(
            lambda src: tf.cast(self.src_vocab_table.lookup(src), tf.int32))

        # Add in the data counts
        infer_dataset_ids = infer_dataset_ids.map(
            lambda src: (src, tf.size(src)))

        def batching_func(x):
            """Pad the source sequences with eos tokens."""
            return x.padded_batch(
                batch_size,
                padded_shapes=(tf.TensorShape([None]),
                               tf.TensorShape([])),
                padding_values=(self.hparams.eos_id, 0)
            )

        infer_dataset_ids = batching_func(infer_dataset_ids)

        infer_iter = infer_dataset_ids.make_initializable_iterator()
        (src_ids, src_seq_len) = infer_iter.get_next()

        return BatchedInput(initializer=infer_iter.initializer,
                            source=src_ids,
                            target_input=None,
                            target_output=None,
                            source_sequence_length=src_seq_len,
                            target_sequence_length=None)

    def _load_dataset(self, dataset_dir):
        src_dataset = tf.data.TextLineDataset(
            [os.path.join(dataset_dir, 'train.enc')])
        tgt_dataset = tf.data.TextLineDataset(
            [os.path.join(dataset_dir, 'train.dec')])
        src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        self.dataset = src_tgt_dataset

    def _convert_to_tokens(self):

        # Ignore the empty lines
        self.dataset = self.dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src) > 0,
                                            tf.size(tgt) > 0))

        self.dataset = self.dataset.map(
            lambda src, tgt: (tf.string_split([src]).values,
                              tf.string_split([tgt]).values))

        # Splice the data up to the max_len, then ignore the rest.
        self.dataset = self.dataset.map(
            lambda src, tgt: (src[:self.src_max_len], tgt[:self.tgt_max_len]))

        # Convert the strings to ids. id = position in vocab table
        self.dataset_ids = self.dataset.map(
            lambda src, tgt: (tf.cast(self.src_vocab_table.lookup(src),
                                      tf.int32),
                              tf.cast(self.tgt_vocab_table.lookup(tgt),
                                      tf.int32)))


def check_vocab(vocab_file):
    """Check and get the vocab"""
    if tf.gfile.Exists(vocab_file):
        vocab_list = []
        with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file, 'rb')) as f:
            for word in f:
                vocab_list.append(word.strip())
    else:
        raise FileNotFoundError(vocab_file + " not found")

    return len(vocab_list), vocab_list


def create_vocab_tables(src_vocab_file, tgt_vocab_file, unk_id):
    """Create the vocab lookup table"""
    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file,
                                                       default_value=unk_id)
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file,
                                                       default_value=unk_id)

    return src_vocab_table, tgt_vocab_table


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ["initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"])):
    pass
