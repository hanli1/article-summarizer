import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import embedding_ops
import random
import nltk
import pickle
import utils
import preprocess


class EncoderDecoderSummarizer:
    PERIOD_TOKEN = "<PERIOD_TOKEN>"
    DECODER_START_TOKEN = "<DECODER_START_TOKEN>"
    ZERO_TOKEN = "<ZERO_TOKEN>"
    DEFAULT_UNKNOWN_TOKEN = "<DEFAULT_UNKNOWN_TOKEN>"

    def __init__(self,
                 training_set,
                 validation_set,
                 embedding_file,
                 embedding_size,
                 num_hidden_units,
                 learning_rate,
                 num_epochs,
                 batch_size,
                 dropout_rate,
                 max_gradient_norm):
        """
        Initialization of some data sources for the neural network and also the hyperparameters
        """
        self.training_set = training_set
        self.validation_set = validation_set
        self.embedding_file = embedding_file
        self.lexicon = set()
        self.lexicon_word_to_id = {}
        self.lexicon_id_to_word = {}
        self.embeddings = {}
        self.unknown_words = set()
        self.encoder_input_data_placeholder = None
        self.embedding_encoder = None
        self.encoder_sequence_lengths_placeholder = None
        self.encoder_embeddings_placeholder = None
        self.dropout_placeholder = None
        self.decoder_input_data_placeholder = None
        self.decoder_embeddings_placeholder = None
        self.decoder_sequence_lengths_placeholder = None
        self.decoder_outputs_placeholder = None
        self.target_weights_placeholder = None
        self.embedding_size = embedding_size
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.max_gradient_norm = max_gradient_norm
        self.model_output_file = "./encoder_decoder_summarizer_model"

        self.initialize_lexicon()
        with open("lexicon_word_to_id.pickle", "wb") as f1:
            pickle.dump(self.lexicon_word_to_id, f1)
        with open("lexicon_id_to_word.pickle", "wb") as f2:
            pickle.dump(self.lexicon_id_to_word, f2)

        self.initialize_embeddings()
        special_tokens = [EncoderDecoderSummarizer.PERIOD_TOKEN, EncoderDecoderSummarizer.DECODER_START_TOKEN,
                          EncoderDecoderSummarizer.ZERO_TOKEN, EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN]
        with open("special_tokens_embeddings.pickle", "wb") as f3:
            special_tokens_embeddings = {}
            for special_token in special_tokens:
                special_tokens_embeddings[special_token] = self.embeddings[special_token]
            pickle.dump(special_tokens_embeddings, f3)
        with open("unknown_tokens_embeddings.pickle", "wb") as f4:
            unknown_tokens_embeddings = {}
            for unknown_word in self.unknown_words:
                unknown_tokens_embeddings[unknown_word] = self.embeddings[unknown_word]
            pickle.dump(unknown_tokens_embeddings, f4)

    def model_computation_graph(self, training=True, inference_size=1):
        self.embedding_encoder = tf.placeholder(tf.float32, shape=(len(self.lexicon), self.embedding_size))
        self.encoder_input_data_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.encoder_embeddings_placeholder = embedding_ops.embedding_lookup(self.embedding_encoder,
                                                                             self.encoder_input_data_placeholder)
        self.encoder_sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,))
        #self.dropout_placeholder = tf.placeholder(tf.float32)

        if training:
            self.decoder_input_data_placeholder = tf.placeholder(tf.int32, shape=(None, None))
            self.decoder_embeddings_placeholder = embedding_ops.embedding_lookup(self.embedding_encoder,
                                                                                 self.decoder_input_data_placeholder)
            self.decoder_sequence_lengths_placeholder = tf.placeholder(tf.int32, shape=(None,))
            self.decoder_outputs_placeholder = tf.placeholder(tf.int32, shape=(None, None))
            self.target_weights_placeholder = tf.placeholder(tf.float32, shape=(None, None))

        # Encoder stage
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_embeddings_placeholder,
                                                           sequence_length=self.encoder_sequence_lengths_placeholder,
                                                           time_major=True, dtype=tf.float32)

        # Decoder stage
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden_units)
        projection_layer = layers_core.Dense(len(self.lexicon), use_bias=False)

        if training:
            # Train Decoder LSTM when given the desired sequences of words
            helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embeddings_placeholder,
                                                       self.decoder_sequence_lengths_placeholder, time_major=True)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
                                                      output_layer=projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True)
            logits = outputs.rnn_output

            # Loss function
            cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs_placeholder,
                                                                      logits=logits)
            train_loss = tf.reduce_mean(tf.reduce_sum(cross_ent * self.target_weights_placeholder))

            # Gradient Computation and Learning
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
            return (train_loss, update_step)
        else:
            # Use Decoder LSTM to predict sequences of words
            batch_size = self.encoder_input_data_placeholder.shape[1].value
            decoder_start_token_ids = tf.fill([tf.shape(self.encoder_input_data_placeholder)[1]],
                                              self.lexicon_word_to_id[EncoderDecoderSummarizer.DECODER_START_TOKEN])
            period_token_id = self.lexicon_word_to_id[EncoderDecoderSummarizer.PERIOD_TOKEN]
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_encoder, decoder_start_token_ids,
                                                              period_token_id)
            current_state = encoder_state
            word_ids = None

            for i in range(inference_size):
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, current_state,
                                                          output_layer=projection_layer)
                maximum_iterations = tf.to_int32(tf.round(tf.to_float(tf.reduce_max(
                    self.encoder_sequence_lengths_placeholder)) * 0.5))
                outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                              output_time_major=True,
                                                                              swap_memory=True,
                                                                              maximum_iterations=maximum_iterations)
                current_word_ids = outputs.sample_id
                if word_ids is None:
                    word_ids = current_word_ids
                else:
                    word_ids = tf.concat([word_ids, current_word_ids], 0)
                current_state = decoder_state

            return word_ids

    def get_embedding_for_token(self, token):
        if (token.lower()) not in self.embeddings:
            self.embeddings[token.lower()] = self.RS.multivariate_normal(self.m, np.diag(self.v))
        embedding = self.embeddings.get((token).lower())
        return embedding

    def initialize_lexicon(self):
        try:
            with open("lexicon_word_to_id.pickle", "rb") as f1:
                self.lexicon_word_to_id = pickle.load(f1)
            with open("lexicon_id_to_word.pickle", "rb") as f2:
                self.lexicon_id_to_word = pickle.load(f2)
            if EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN not in self.lexicon_word_to_id:
                self.lexicon_word_to_id[EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN] = len(self.lexicon_word_to_id)
                self.lexicon_id_to_word[len(self.lexicon_word_to_id)] = EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN
        except Exception:
            self.lexicon_word_to_id = {}
            self.lexicon_id_to_word = {}

        if (len(self.lexicon_word_to_id) == 0) or (len(self.lexicon_id_to_word) == 0):
            self.lexicon_word_to_id = {}
            self.lexicon_id_to_word = {}
            current_word_id = 0
            self.lexicon = set()
            special_tokens = [EncoderDecoderSummarizer.PERIOD_TOKEN, EncoderDecoderSummarizer.DECODER_START_TOKEN,
                              EncoderDecoderSummarizer.ZERO_TOKEN, EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN]
            for special_token in special_tokens:
                self.lexicon.add(special_token)
                self.lexicon_word_to_id[special_token] = current_word_id
                self.lexicon_id_to_word[current_word_id] = special_token
                current_word_id += 1

            texts = []
            for (article, summary) in (self.training_set + self.validation_set):
                texts.append(article)
                texts.append(summary)

            for text in texts:
                text_tokens = list(map(lambda x: x.lower(), nltk.word_tokenize(text)))
                for text_token in text_tokens:
                    if text_token not in self.lexicon:
                        self.lexicon.add(text_token)
                        self.lexicon_word_to_id[text_token] = current_word_id
                        self.lexicon_id_to_word[current_word_id] = text_token
                        current_word_id += 1
        else:
            for word in self.lexicon_word_to_id:
                self.lexicon.add(word)
        print("Lexicon Size:{}".format(len(self.lexicon)))

    def initialize_embeddings(self):
        self.embeddings = {}
        with open(self.embedding_file, 'r') as f:
            for line in f.readlines():
                line_components = line.split()
                if line_components[0] in self.lexicon:
                    word = line_components[0]
                    word_embedding = line_components[1:(self.embedding_size + 1)]
                    self.embeddings[word] = np.array([float(i) for i in word_embedding])

        wvecs = []
        for item in self.embeddings.items():
            wvecs.append(item[1])
        s = np.vstack(wvecs)

        # Gather the distribution hyperparameters
        self.v = np.var(s, 0)
        self.m = np.mean(s, 0)
        self.RS = np.random.RandomState()
        special_tokens = [EncoderDecoderSummarizer.PERIOD_TOKEN, EncoderDecoderSummarizer.DECODER_START_TOKEN,
                          EncoderDecoderSummarizer.ZERO_TOKEN, EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN]
        try:
            with open("special_tokens_embeddings.pickle", "rb") as f1:
                special_tokens_embeddings = pickle.load(f1)
                for special_token in special_tokens_embeddings:
                    self.embeddings[special_token] = special_tokens_embeddings[special_token]
        except Exception:
            for special_token in special_tokens:
                self.embeddings[special_token] = self.RS.multivariate_normal(self.m, np.diag(self.v))

        try:
            with open("unknown_tokens_embeddings.pickle", "rb") as f2:
                unknown_tokens_embeddings = pickle.load(f2)
                for unknown_token in unknown_tokens_embeddings:
                    self.unknown_words.add(unknown_token)
                    self.embeddings[unknown_token] = unknown_tokens_embeddings[unknown_token]
        except Exception:
            for token in self.lexicon:
                if token not in self.embeddings:
                    self.unknown_words.add(token)
                    self.embeddings[token] = self.RS.multivariate_normal(self.m, np.diag(self.v))

    def get_embedding_encoder(self):
        embedding_list = []
        for word in self.lexicon:
            current_embedding = self.embeddings[word]
            embedding_list.append(current_embedding)
        embedding_encoder = np.array(embedding_list)
        return embedding_encoder

    def get_token_ids_from_text(self, text):
        text_sentences = nltk.sent_tokenize(text)
        text_token_ids = []
        special_tokens = [EncoderDecoderSummarizer.PERIOD_TOKEN, EncoderDecoderSummarizer.DECODER_START_TOKEN,
                          EncoderDecoderSummarizer.ZERO_TOKEN, EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN]
        for text_sentence in text_sentences:
            sentence_tokens = nltk.word_tokenize(text_sentence)
            if sentence_tokens[-1] == ".":
                sentence_tokens[-1] = EncoderDecoderSummarizer.PERIOD_TOKEN
            else:
                sentence_tokens.append(EncoderDecoderSummarizer.PERIOD_TOKEN)

            sentence_token_ids = []
            for sentence_token in sentence_tokens:
                if sentence_token not in special_tokens:
                    try:
                        sentence_token_ids.append(self.lexicon_word_to_id[sentence_token.lower()])
                    except Exception:
                        sentence_token_ids.append(self.lexicon_word_to_id[EncoderDecoderSummarizer.DEFAULT_UNKNOWN_TOKEN])
                else:
                    sentence_token_ids.append(self.lexicon_word_to_id[sentence_token])
            text_token_ids = text_token_ids + sentence_token_ids

        return text_token_ids

    def get_batch_sequence_data(self, sequence_list_token_ids):
        max_sequence_size = max([len(s) for s in sequence_list_token_ids])
        sequence_input_data = None
        sequence_lengths = np.array([], dtype=np.int32)
        for sequence_token_ids in sequence_list_token_ids:
            current_token_ids = sequence_token_ids
            sequence_lengths = np.append(sequence_lengths, len(current_token_ids))
            for i in range(len(current_token_ids), max_sequence_size):
                current_token_ids.append(self.lexicon_word_to_id[EncoderDecoderSummarizer.ZERO_TOKEN])
            current_token_ids = np.array(current_token_ids)
            current_token_ids = current_token_ids.reshape(1, max_sequence_size)
            if sequence_input_data is None:
                sequence_input_data = current_token_ids
            else:
                sequence_input_data = np.append(sequence_input_data, current_token_ids, axis=0)
        return (np.transpose(sequence_input_data), sequence_lengths)

    def get_batch_encoder_data(self, batch_articles_list):
        article_list_token_ids = []

        for article in batch_articles_list:
            article_token_ids = self.get_token_ids_from_text(article)
            article_list_token_ids.append(article_token_ids)

        encoder_input_data, encoder_sequence_lengths = self.get_batch_sequence_data(article_list_token_ids)

        return (encoder_input_data, encoder_sequence_lengths)

    def get_batch_decoder_data(self, batch_summaries_list):
        summary_list_token_output_ids = []
        summary_list_token_input_ids = []

        for summary in batch_summaries_list:
            summary_token_output_ids = self.get_token_ids_from_text(summary)
            summary_list_token_output_ids.append(summary_token_output_ids)

            summary_token_input_ids = [self.lexicon_word_to_id[EncoderDecoderSummarizer.DECODER_START_TOKEN]]
            for i in range(0, len(summary_token_output_ids) - 1):
                summary_token_input_ids.append(summary_token_output_ids[i])
            summary_list_token_input_ids.append(summary_token_input_ids)

        decoder_outputs, decoder_sequence_lengths = self.get_batch_sequence_data(summary_list_token_output_ids)
        decoder_input_data, decoder_sequence_lengths = self.get_batch_sequence_data(summary_list_token_input_ids)
        target_weights = None
        for i in range(decoder_outputs.shape[0]):
            one_zero_row = []
            for j in range(len(decoder_outputs[i])):
                if decoder_outputs[i][j] == (self.lexicon_word_to_id[EncoderDecoderSummarizer.ZERO_TOKEN]):
                    one_zero_row.append(0.0)
                else:
                    one_zero_row.append(1.0)
            one_zero_row = np.array(one_zero_row)
            one_zero_row = one_zero_row.reshape(1, decoder_outputs.shape[1])
            if target_weights is None:
                target_weights = one_zero_row
            else:
                target_weights = np.append(target_weights, one_zero_row, axis=0)

        return (decoder_input_data, decoder_sequence_lengths, decoder_outputs, target_weights)

    def train(self):
        with tf.Graph().as_default():
            train_loss, update_step = self.model_computation_graph(training=True)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                embedding_encoder = self.get_embedding_encoder()
                for i in range(self.num_epochs):
                    # Randomly shuffle model data for each epoch
                    random.shuffle(self.training_set)
                    loss_value = None
                    for j in range(0, len(self.training_set), self.batch_size):
                        # Get all input sentences and labels for current batch
                        batch_articles_and_summaries_list = self.training_set[j:(j + self.batch_size)]
                        batch_articles_list = list(map(lambda x: x[0], batch_articles_and_summaries_list))
                        batch_summaries_list = list(map(lambda x: x[1], batch_articles_and_summaries_list))
                        encoder_input_data, encoder_sequence_lengths = self.get_batch_encoder_data(batch_articles_list)
                        decoder_input_data, decoder_sequence_lengths, decoder_outputs, target_weights = \
                            self.get_batch_decoder_data(batch_summaries_list)
                        # Perform a weights/variables update with this batch's input data and output data
                        feed_dict = {
                            self.embedding_encoder: embedding_encoder,
                            self.encoder_input_data_placeholder: encoder_input_data,
                            self.encoder_sequence_lengths_placeholder: encoder_sequence_lengths,
                            self.decoder_input_data_placeholder: decoder_input_data,
                            self.decoder_sequence_lengths_placeholder: decoder_sequence_lengths,
                            self.decoder_outputs_placeholder: decoder_outputs,
                            self.target_weights_placeholder: target_weights
                        }
                        _, loss_value = session.run([update_step, train_loss], feed_dict=feed_dict)
                        print("Completed Batch {} of Epoch {}; Loss Value: {}".format((j // self.batch_size) + 1,
                            i + 1, loss_value))
                        if j >= 100 and j % 100 == 0:
                            saver.save(session, self.model_output_file)
                    print("Completed Epoch {} of {}; Loss Value: {}".format(i + 1, self.num_epochs, loss_value))
                    saver.save(session, self.model_output_file)

    def predict(self, articles, inference_size=1):
        res = []

        with tf.Graph().as_default():
            word_ids_op = self.model_computation_graph(False, inference_size)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                saver.restore(session, self.model_output_file)
                embedding_encoder = self.get_embedding_encoder()
                encoder_input_data, encoder_sequence_lengths = self.get_batch_encoder_data(articles)
                feed_dict = {
                    self.embedding_encoder: embedding_encoder,
                    self.encoder_input_data_placeholder: encoder_input_data,
                    self.encoder_sequence_lengths_placeholder: encoder_sequence_lengths
                }
                prediction_ids = session.run(word_ids_op, feed_dict=feed_dict)
                print("PREDICTION: {}".format(" ".join(list(map(lambda x: self.lexicon_id_to_word[x[0]],
                                                                prediction_ids)))))


if __name__ == '__main__':
    all_data = utils.get_kaggle_data()
    random.shuffle(all_data)
    training_size = int(len(all_data) * 0.9)
    encoder_decoder_summarizer = EncoderDecoderSummarizer(training_set=all_data[:training_size],
                                                          validation_set=all_data[training_size:],
                                                          embedding_file="glove/glove.6B.300d.txt",
                                                          embedding_size=300,
                                                          num_hidden_units=256,
                                                          learning_rate=0.001,
                                                          num_epochs=20,
                                                          batch_size=10,
                                                          dropout_rate=0.33,
                                                          max_gradient_norm=5)
