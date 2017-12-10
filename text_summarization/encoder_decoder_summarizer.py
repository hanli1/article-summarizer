import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import embedding_ops
import random
import nltk
import pickle

class EncoderDecoderSummarizer:
    PERIOD_TOKEN = "<PERIOD_TOKEN>"
    DECODER_START_TOKEN = "<DECODER_START_TOKEN>"
    ZERO_TOKEN = "<ZERO_TOKEN>"

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
            self.target_weights_placeholder = tf.placeholder(tf.int32, shape=(None, None))

        # Encoder stage
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_embeddings_placeholder,
                                                           sequence_length=self.encoder_sequence_lengths_placeholder,
                                                           time_major=True)

        # Decoder stage
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden_units)
        projection_layer = layers_core.Dense(len(self.lexicon), use_bias=False)

        if training:
            # Train Decoder LSTM when given the desired sequences of words
            helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embeddings_placeholder,
                                                       self.decoder_sequence_lengths_placeholder, time_major=True)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,
                                                      output_layer=projection_layer)
            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True)
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
            decoder_start_token_ids = tf.fill([batch_size],
                                              self.lexicon_word_to_id[EncoderDecoderSummarizer.DECODER_START_TOKEN])
            period_token_id = self.lexicon_word_to_id[EncoderDecoderSummarizer.PERIOD_TOKEN]
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_encoder, decoder_start_token_ids,
                                                              period_token_id)
            current_state = encoder_state
            word_ids = None

            for i in range(inference_size):
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, current_state,
                                                          output_layer=projection_layer)
                maximum_iterations = tf.round(tf.reduce_max(self.encoder_sequence_lengths_placeholder) * 0.5)
                outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                              maximum_iterations=maximum_iterations)
                word_ids = outputs.sample_id
                print (word_ids)
                print (word_ids.shape)
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
        except Exception:
            self.lexicon_word_to_id = {}
        try:
            with open("lexicon_id_to_word.pickle", "rb") as f2:
                self.lexicon_id_to_word = pickle.load(f2)
        except Exception:
            self.lexicon_id_to_word = {}

        if (len(self.lexicon_word_to_id) == 0) or (len(self.lexicon_id_to_word) == 0):
            self.lexicon_word_to_id = {}
            self.lexicon_id_to_word = {}
            current_word_id = 0
            self.lexicon = set()
            special_tokens = [EncoderDecoderSummarizer.PERIOD_TOKEN, EncoderDecoderSummarizer.DECODER_START_TOKEN,
                              EncoderDecoderSummarizer.ZERO_TOKEN]
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
        with open(self.embedding_file, 'r', encoding="utf8") as f:
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
                          EncoderDecoderSummarizer.ZERO_TOKEN]
        try:
            with open("special_tokens_embeddings.pickle", "rb") as f1:
                special_tokens_embeddings = pickle.load(f1)
                for special_token in special_tokens_embeddings:
                    self.embeddings[special_token] = special_tokens_embeddings[special_token]
        except Exception:
            for special_token in special_tokens:
                self.embeddings[special_token] = self.RS.multivariate_normal(self.m, np.diag(self.v))

    def get_embedding_encoder(self):
        pass

    def get_batch_encoder_and_decoder_data(self, batch_articles_list):
        pass

    def train(self):
        self.initialize_lexicon()
        with open("lexicon_word_to_id.pickle", "wb") as f1:
            pickle.dump(self.lexicon_word_to_id, f1)
        with open("lexicon_id_to_word.pickle", "wb") as f2:
            pickle.dump(self.lexicon_id_to_word, f2)

        self.initialize_embeddings()
        special_tokens = [EncoderDecoderSummarizer.PERIOD_TOKEN, EncoderDecoderSummarizer.DECODER_START_TOKEN,
                          EncoderDecoderSummarizer.ZERO_TOKEN]
        with open("special_tokens_embeddings.pickle", "wb") as f3:
            special_tokens_embeddings = {}
            for special_token in special_tokens:
                special_tokens_embeddings[special_token] = self.embeddings[special_token]
            pickle.dump(special_tokens_embeddings, f3)

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
                        batch_articles_list = self.training_set[j:(j + self.batch_size)]
                        batch_encoder_and_decoder_data = self.get_batch_encoder_and_decoder_data(batch_articles_list)
                        # Perform a weights/variables update with this batch's input data and output data
                        feed_dict = {
                            self.embedding_encoder: embedding_encoder,
                            self.encoder_input_data_placeholder: batch_encoder_and_decoder_data[0],
                            self.encoder_sequence_lengths_placeholder: batch_encoder_and_decoder_data[1],
                            self.decoder_input_data_placeholder: batch_encoder_and_decoder_data[2],
                            self.decoder_sequence_lengths_placeholder: batch_encoder_and_decoder_data[3],
                            self.decoder_outputs_placeholder: batch_encoder_and_decoder_data[4],
                            self.target_weights_placeholder: batch_encoder_and_decoder_data[5]
                        }
                        _, loss_value = session.run([update_step, train_loss], feed_dict=feed_dict)
                        print("Completed Batch {} of Epoch {}; Loss Value: {}".format((j // self.batch_size) + 1,
                            i + 1, loss_value))
                    print("Completed Epoch {} of {}; Loss Value: {}".format(i + 1, self.num_epochs, loss_value))
                saver.save(session, self.model_output_file)

    def predict(self, possible_test_tuples):
        res = []
        prediction = None

        with tf.Graph().as_default():
            predict_id_op = self.feed_forward_computation_graph(True)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                saver.restore(session, self.model_output_file)
                valid_tuples, input_data, label_data = self.get_feature_and_output_vectors(possible_test_tuples)
                feed_dict = {
                    self.input_data_placeholder: input_data,
                    self.dropout_placeholder: 0.0
                }
                prediction_ids = session.run(predict_id_op, feed_dict=feed_dict)
            self.num_validation_tuples_predicted += 1
            print("Num Test Examples Predicted: {}".format(self.num_validation_tuples_predicted))

        return (valid_tuples, input_data, label_data, prediction_ids)