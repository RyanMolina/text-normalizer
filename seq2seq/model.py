import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class ModelBuilder(object):
    def __init__(self, dataset, model_dir, batch_input, training=True):
        self.model_dir = model_dir
        self.training = training
        self.batch_input = batch_input
        self.src_vocab_table = dataset.src_vocab_table
        self.tgt_vocab_table = dataset.tgt_vocab_table
        self.reverse_vocab_table = dataset.reverse_vocab_table

        self.src_vocab_size = dataset.src_vocab_size
        self.tgt_vocab_size = dataset.tgt_vocab_size

        self.hparams = dataset.hparams

        self.num_layers = self.hparams.num_layers
        self.time_major = self.hparams.time_major

        initializer = get_initializer(self.hparams.init_op,
                                      self.hparams.random_seed,
                                      self.hparams.init_weight)

        tf.get_variable_scope().set_initializer(initializer)

        self.embedding_encoder, self.embedding_decoder = (
            create_enc_dec_embedding(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=self.hparams.num_units,
                tgt_embed_size=self.hparams.num_units))
        self.batch_size = tf.size(self.batch_input.source_sequence_length)

        with tf.variable_scope("build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(self.src_vocab_size,
                                                      use_bias=False,
                                                      name="output_projection")

        print("+ Building graph for the seq2seq ...")
        res = self.build_graph(self.hparams)

        if self.training:
            self.train_loss = res[1]
            self.word_count = \
                tf.reduce_sum(self.batch_input.source_sequence_length) \
                + tf.reduce_sum(self.batch_input.target_sequence_length)

            self.predict_count = \
                tf.reduce_sum(self.batch_input.target_sequence_length)
        else:
            (self.infer_logits,
             _, self.final_context_state,
             self.sample_id) = res

            self.sample_words = \
                self.reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        if self.training:
            self.learning_rate = tf.constant(self.hparams.learning_rate)
            self.learning_rate = self._get_learning_rate_decay(self.hparams)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.train_loss, params)

            clipped_gradients, gradient_norm_summary, grad_norm = \
                gradient_clip(gradients,
                              max_gradient_norm=self.hparams.max_gradient_norm)

            self.grad_norm = grad_norm

            self.update = opt.apply_gradients(zip(clipped_gradients, params),
                                              global_step=self.global_step)

            self.train_summary = tf.summary.merge(
                [tf.summary.scalar("learning_rate", self.learning_rate),
                 tf.summary.scalar("train_loss", self.train_loss)]
                + gradient_norm_summary)
        else:
            self.infer_summary = self._get_infer_summary()

        self.saver = tf.train.Saver(tf.global_variables())

    def _get_learning_rate_decay(self, hparams):
        start_decay_step = hparams.start_decay_step
        decay_steps = hparams.decay_steps
        decay_factor = hparams.decay_factor

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def train(self, sess):
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size,
                         self.grad_norm,
                         self.learning_rate])

    def build_graph(self, hparams):
        dtype = tf.float32

        with tf.variable_scope("dynamic_seq2seq", dtype=dtype):
            encoder_outputs, encoder_state = self._build_encoder()

            logits, sample_id, final_context_state = \
                self._build_decoder(encoder_outputs, encoder_state)

            if self.training:
                loss = self._compute_loss(logits)
            else:
                loss = None
            return logits, loss, final_context_state, sample_id

    def _build_encoder(self):
        source = self.batch_input.source

        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder,
                                                     source)

            cell = self._build_encoder_cell()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell,
                encoder_emb_inp,
                dtype=scope.dtype,
                sequence_length=self.batch_input.source_sequence_length,
                time_major=self.time_major,
                swap_memory=True)

            return encoder_outputs, encoder_state

    def _build_decoder(self, encoder_outputs, encoder_state):

        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(
            tf.constant(self.hparams.sos_token)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(
            tf.constant(self.hparams.eos_token)), tf.int32)

        if self.hparams.tgt_max_len_infer:
            maximum_iterations = self.hparams.tgt_max_len_infer
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(
                self.batch_input.source_sequence_length)
            maximum_iterations = tf.to_int32(
                tf.round(tf.to_float(max_encoder_length)
                         * decoding_length_factor))

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                encoder_outputs, encoder_state,
                self.batch_input.source_sequence_length)

            # Train
            if self.training:
                # decoder_emp_inp: [max_time, batch_size, num_units]
                target_input = self.batch_input.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, target_input)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, self.batch_input.target_sequence_length,
                    time_major=self.time_major)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state, )

                # Dynamic decoding
                outputs, final_context_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                        my_decoder,
                        output_time_major=self.time_major,
                        swap_memory=True,
                        scope=decoder_scope)

                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)

            # Inference
            else:
                # length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                # Helper
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embedding_decoder, start_tokens, end_token)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state,
                    output_layer=self.output_layer  # applied per timestep
                )

                # Dynamic decoding
                outputs, final_context_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                        my_decoder,
                        maximum_iterations=maximum_iterations,
                        output_time_major=self.time_major,
                        swap_memory=True,
                        scope=decoder_scope)

                logits = outputs.rnn_output
                sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def _build_encoder_cell(self):
        """Build a multi-layer RNN cell that can be used by encoder."""
        return create_rnn_cell(
            num_units=self.hparams.num_units,
            num_layers=self.hparams.num_layers,
            input_keep_prob=self.hparams.input_keep_prob,
            output_keep_prob=self.hparams.output_keep_prob)

    def _build_decoder_cell(self, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build a RNN cell with attention mechanism
        that can be used by decoder.
        """

        num_units = self.hparams.num_units
        num_layers = self.hparams.num_layers

        dtype = tf.float32

        # Ensure memory is batch-major
        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        batch_size = self.batch_size

        # Create the attention mechanism
        if self.hparams.attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length)
        elif self.hparams.attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                scale=True)
        elif self.hparams.attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length)
        elif self.hparams.attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                normalize=True)
        else:
            raise ValueError("Unknown attention option {}"
                             .format(self.hparams.attention_option))

        cell = create_rnn_cell(
            num_units=num_units,
            num_layers=num_layers,
            input_keep_prob=self.hparams.input_keep_prob,
            output_keep_prob=self.hparams.output_keep_prob)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (not self.training)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=num_units,
            alignment_history=alignment_history,
            name="attention")

        if self.hparams.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
                cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.batch_input.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.batch_input.target_sequence_length,
            max_time,
            dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def infer(self, sess):

        infer_logits, infer_summary, sample_id, sample_words = sess.run([
            self.infer_logits,
            self.infer_summary,
            self.sample_id,
            self.sample_words
        ])

        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            sample_words = sample_words.transpose()
        return sample_words, infer_summary

    def _get_infer_summary(self):
        return self.final_context_state.alignment_history.stack()


def get_initializer(init_op, seed=None, init_weight=None):
    if init_op == "uniform":
        return tf.random_uniform_initializer(-init_weight,
                                             init_weight,
                                             seed=seed)
    elif init_op == "glorot_normal":
        return tf.contrib.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op {0}".format(init_op))


def create_enc_dec_embedding(src_vocab_size,
                             tgt_vocab_size,
                             src_embed_size,
                             tgt_embed_size,
                             dtype=tf.float32,
                             num_partitions=0):
    if num_partitions <= 1:
        partitioner = None
    else:
        partitioner = tf.fixed_size_paritioner(num_partitions)

    with tf.variable_scope("embeddings",
                           dtype=dtype,
                           partitioner=partitioner):
        with tf.variable_scope("encoder", partitioner=partitioner):
            embedding_encoder = tf.get_variable("embedding_encoder",
                                                [src_vocab_size,
                                                 src_embed_size],
                                                dtype)
        with tf.variable_scope("decoder", partitioner=partitioner):
            embedding_decoder = tf.get_variable("embedding_decoder",
                                                [tgt_vocab_size,
                                                 tgt_embed_size],
                                                dtype)

    return embedding_encoder, embedding_decoder


def _single_cell(num_units,
                 input_keep_prob,
                 output_keep_prob,
                 device_str=None):
    single_cell = tf.contrib.rnn.GRUCell(num_units)

    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell,
        input_keep_prob=input_keep_prob,
        output_keep_prob=output_keep_prob)
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

    return single_cell


def create_rnn_cell(num_units, num_layers, input_keep_prob, output_keep_prob):
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(
            num_units, input_keep_prob, output_keep_prob)
        cell_list.append(single_cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm),
                             tf.summary.scalar(
                                 "clipped_gradient",
                                 tf.global_norm(clipped_gradients))]

    return clipped_gradients, gradient_norm_summary, gradient_norm


def load_model(model, ckpt, session):
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    return model


def create_or_load(model, session):
    latest_ckpt = tf.train.latest_checkpoint(model.model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session)
    else:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

    global_step = model.global_step.eval(session=session)
    return model, global_step
