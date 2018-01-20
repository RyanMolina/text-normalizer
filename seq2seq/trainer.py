"""This module contains the Trainer class"""
import os
import random
import time
import math
import tensorflow as tf
from seq2seq import model, dataset, utils

"""
train loss value
add accuracy evaluation per interval
cross entropy sequence loss
accuracy chart
* perplexity
change the decay scheme to
decay factor of 10 if the validation loss stopped.
70/20/10
train/test/val

add reverse seq
"""


class Trainer(object):
    def __init__(self, data_dir, model_dir, hparams=None):
        self.train_graph = tf.Graph()

        with self.train_graph.as_default(), tf.container("train"):
            self.train_dataset = dataset.Dataset(dataset_dir=data_dir,
                                                 hparams=hparams)

            self.train_batch = self.train_dataset.get_training_batch()

            self.skip_count_placeholder = tf.placeholder(shape=(),
                                                         dtype=tf.int64)

            self.train_model = model.ModelBuilder(training=True,
                                                  dataset=self.train_dataset,
                                                  model_dir=model_dir,
                                                  batch_input=self.train_batch)

        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default(), tf.container("infer"):
            self.infer_dataset = dataset.Dataset(dataset_dir=data_dir,
                                                 hparams=hparams,
                                                 training=False)
            self.src_placeholder = tf.placeholder(shape=[None],
                                                  dtype=tf.string)

            self.src_dataset = tf.data.Dataset.from_tensor_slices(
                self.src_placeholder)

            self.batch_size_placeholder = tf.placeholder(shape=[],
                                                         dtype=tf.int64)

            self.infer_batch = self.infer_dataset.get_inference_batch(
                self.src_dataset, self.batch_size_placeholder)

            self.infer_model = model.ModelBuilder(training=False,
                                                  dataset=self.infer_dataset,
                                                  model_dir=model_dir,
                                                  batch_input=self.infer_batch)
        self.model_dir = model_dir
        self.hparams = self.train_dataset.hparams

    def train(self, target=""):
        log_device_placement = self.hparams.log_device_placement
        num_train_steps = self.hparams.num_train_steps
        steps_per_stats = self.hparams.steps_per_stats
        save_interval = self.hparams.save_interval

        config_proto = tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=True)

        config_proto.gpu_options.allow_growth = True

        train_sess = tf.Session(target=target,
                                config=config_proto,
                                graph=self.train_graph)

        infer_sess = tf.Session(target=target,
                                config=config_proto,
                                graph=self.infer_graph)

        with self.train_graph.as_default():
            self.train_model, global_step = model.create_or_load(
                self.train_model, train_sess)

            train_sess.run(self.train_batch.initializer)

        # Summary Writer
        summary_name = "train_log"
        summary_writer = tf.summary.FileWriter(os.path.join(
            self.train_model.model_dir, summary_name), self.train_graph)

        # Log and output files
        log_file = os.path.join(self.train_model.model_dir,
                                "log_{:d}".format(int(time.time())))

        log_f = tf.gfile.GFile(log_file, mode="a")
        utils.print_out("# log_files={}".format(log_file), log_f)

        # First Evaluation

        last_stats_step = global_step
        last_save_step = global_step

        stats = self.init_stats()
        train_perp = 0.0

        utils.print_out(
            "# Start step {}, lr {:.2e}, {}"
            .format(global_step,
                    self.train_model.learning_rate.eval(session=train_sess),
                    time.ctime()),
            log_f)

        skip_count = self.hparams.batch_size * self.hparams.epoch_step
        utils.print_out(
            "# Init train iterator, skipping {} elements".format(skip_count))

        train_sess.run(
            self.train_batch.initializer,
            feed_dict={self.skip_count_placeholder: skip_count})

        while global_step < num_train_steps:
            train_start_time = time.time()
            try:
                step_result = self.train_model.train(train_sess)
                self.hparams.epoch_step += 1
            except tf.errors.OutOfRangeError:
                self.hparams.epoch_step = 0

                utils.print_out(
                    "# Finished an epoch, step {}. Perform external evaluation"
                    .format(global_step))

                self.run_sample_decode(self.infer_model, infer_sess,
                                       self.hparams, summary_writer,
                                       self.infer_dataset.sample_src_data,
                                       self.infer_dataset.sample_tgt_data)

                # dev_scores, test_scores, _ = self.run_external_eval(
                #     self.infer_model, infer_sess, self.model_dir,
                #     self.hparams, summary_writer)
                #
                train_sess.run(
                    self.train_batch.initializer,
                    feed_dict={self.skip_count_placeholder: 0})
                continue
            # Write step summary
            global_step = self.update_stats(stats,
                                            summary_writer,
                                            train_start_time,
                                            step_result)

            if global_step - last_stats_step >= steps_per_stats:
                last_stats_step = global_step
                is_overflow = self.check_stats(stats,
                                               global_step,
                                               steps_per_stats,
                                               log_f)

                if is_overflow:
                    break
                stats = self.init_stats()

            if global_step - last_save_step >= save_interval:
                last_save_step = global_step

                utils.print_out(
                    "# Save eval, global step {}".format(global_step))

                utils.add_summary(summary_writer,
                                  global_step,
                                  "train_ppl",
                                  train_perp)

                # Save checkpoint
                self.train_model.saver.save(
                    train_sess,
                    os.path.join(self.model_dir, "normalize.ckpt"),
                    global_step=global_step)

                # Evaluate on dev/test
                self.run_sample_decode(self.infer_model, infer_sess,
                                       self.hparams, summary_writer,
                                       self.infer_dataset.sample_src_data,
                                       self.infer_dataset.sample_tgt_data)
        self.train_model.saver.save(
            train_sess,
            os.path.join(self.model_dir, "normalize.ckpt"),
            global_step=global_step)

        utils.print_time("# Done train!", train_start_time)

        summary_writer.close()

    def run_sample_decode(self, infer_model, infer_sess, hparams,
                          summary_writer, src_data, tgt_data):
        """Sample decode a random sentence from src_data."""
        with self.infer_graph.as_default():
            loaded_infer_model, global_step = model.create_or_load(
                infer_model, infer_sess)

        self._sample_decode(loaded_infer_model, global_step,
                            infer_sess, hparams,
                            self.infer_batch, src_data, tgt_data,
                            self.src_placeholder,
                            self.batch_size_placeholder, summary_writer)

    @staticmethod
    def _format_results(name, ppl, scores, metrics):
        """Format results."""
        result_str = "{} ppl {:.2f}".format(name, ppl)
        if scores:
            for metric in metrics:
                result_str += ", {} {} {:.1f}".format(
                    name, metric, scores[metric])
        return result_str

    @staticmethod
    def _get_best_results(hparams):
        """Summary of the current best results."""
        return "{}".format((getattr(hparams, "best_accuracy")))

    @staticmethod
    def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                       summary_writer, label):
        """Computing perplexity."""
        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
        ppl = model.compute_perplexity(model, sess, label)
        utils.add_summary(summary_writer, global_step,
                          "{}_ppl".format(label), ppl)
        return ppl

    @staticmethod
    def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                       tgt_data, iterator_src_placeholder,
                       iterator_batch_size_placeholder, summary_writer):
        def get_translation(outputs, tgt_eos):
            """Given batch decoding outputs,
            select a sentence and turn to text.
            """
            if tgt_eos:
                tgt_eos = tgt_eos.encode("utf-8")
            # Select a sentence
            output = outputs.tolist()[0]
            # If there is an eos symbol in outputs, cut them at that point.
            if tgt_eos and tgt_eos in output:
                output = output[:output.index(tgt_eos)]
            return output

        """Pick a sentence and decode."""
        decode_id = random.randint(0, len(src_data) - 1)
        utils.print_out("  # {}".format(decode_id))

        iterator_feed_dict = {
            iterator_src_placeholder: [src_data[decode_id]],
            iterator_batch_size_placeholder: 1,
        }
        sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

        normalizer_outputs, attention_summary = model.infer(sess)

        src_tokens = [t.encode('utf-8') for t in src_data[decode_id].split()]
        replaced = utils.unk_replace(src_tokens, get_translation(
            normalizer_outputs,
            tgt_eos=hparams.eos_token), attention_summary)
        translation = utils.format_text(replaced)

        utils.print_out("    src: {}".format(src_data[decode_id])
                        .replace(' ', '').replace('<space>', ' '))
        utils.print_out("    ref: {}".format(tgt_data[decode_id])
                        .replace(' ', '').replace('<space>', ' '))
        utils.print_out("    nmt: {}".format(translation)
                        .replace(' ', '').replace('<space>', ' '))

        return src_data[decode_id], tgt_data[decode_id], translation

    @staticmethod
    def init_stats():
        """Initialize statistics that we want to keep."""
        return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
                "total_count": 0.0, "grad_norm": 0.0}

    @staticmethod
    def update_stats(stats, summary_writer, start_time, step_result):
        """Update stats: write summary and accumulate statistics."""
        (_, step_loss, step_predict_count, step_summary, global_step,
         step_word_count, batch_size, grad_norm, learning_rate) = step_result

        # Write step summary.
        summary_writer.add_summary(step_summary, global_step)

        # update statistics
        stats["step_time"] += (time.time() - start_time)
        stats["loss"] += (step_loss * batch_size)
        stats["predict_count"] += step_predict_count
        stats["total_count"] += float(step_word_count)
        stats["grad_norm"] += grad_norm
        stats["learning_rate"] = learning_rate

        return global_step

    @staticmethod
    def check_stats(stats, global_step, steps_per_stats, log_f):
        """Print statistics and also check for overflow."""
        # Print statistics for the previous epoch.
        avg_step_time = stats["step_time"] / steps_per_stats
        avg_grad_norm = stats["grad_norm"] / steps_per_stats
        train_ppl = utils.safe_exp(
            stats["loss"] / stats["predict_count"])
        speed = stats["total_count"] / (1000 * stats["step_time"])
        utils.print_out(
            "  global step {} lr {:.2e} "
            "step-time {:.2f}s seq/s {:.2f}K ppl {:.2f}"
            .format(global_step, stats["learning_rate"],
                    avg_step_time, speed, train_ppl), log_f)

        # Check for overflow
        is_overflow = False
        if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
            utils.print_out(
                "  step {} overflow, stop early".format(global_step, log_f))
            is_overflow = True

        return is_overflow
