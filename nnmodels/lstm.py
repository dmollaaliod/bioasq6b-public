"LSTM models"
import tensorflow as tf
import numpy as np
import sqlite3
import random
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial

from sklearn.metrics import mean_squared_error

from nnmodels import compare

EMBEDDINGS = 100
VECTORS = 'allMeSH_2016_%i.vectors.txt' % EMBEDDINGS
DB = 'word2vec_%i.db' % EMBEDDINGS
MAX_NUM_SNIPPETS = 50
SENTENCE_LENGTH = 300

print("LSTM using sentence length=%i" % SENTENCE_LENGTH)
print("LSTM using embedding dimension=%i" % EMBEDDINGS)

with open(VECTORS) as v:
    VOCABULARY = int(v.readline().strip().split()[0]) + 2

if not os.path.exists(DB):
    print("Creating database of vectors %s" % DB)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE vectors (word unicode,
                                       word_index integer,
                                       data unicode)""")
    with open(VECTORS, encoding='utf-8') as v:
        nwords = int(v.readline().strip().split()[0])
        print("Processing %i words" % nwords)
        zeroes = " ".join("0"*EMBEDDINGS)
        # Insert PAD and UNK special words with zeroes
        c.execute("INSERT INTO vectors VALUES (?, ?, ?)", ('PAD', 0, zeroes))
        c.execute("INSERT INTO vectors VALUES (?, ?, ?)", ('UNK', 1, zeroes))
        for i in range(nwords):
            vector = v.readline()
            windex = vector.index(" ")
            w = vector[:windex].strip()
            d = vector[windex:].strip()
            assert len(d.split()) == EMBEDDINGS
            #if i < 5:
            #    print(w)
            #    print(d)
            c.execute("INSERT INTO vectors VALUES (?, ?, ?)", (w, i+2, d))
    c.execute("CREATE INDEX word_idx ON vectors (word)")
    conn.commit()
    conn.close()

#vectordb = sqlite3.connect(DB)

def sentences_to_ids(sentences, sentence_length=SENTENCE_LENGTH):
    """Convert each sentence to a list of word IDs.

Crop or pad to 0 the sentences to ensure equal length if necessary.
Words without ID are assigned ID 1.
>>> sentences_to_ids([['my','first','sentence'],['my','ssecond','sentence'],['yes']], 2)
(([11095, 121], [11095, 1], [21402, 0]), (2, 2, 1))
"""
    return tuple(zip(*map(partial(one_sentence_to_ids,
                                  sentence_length=sentence_length),
                          sentences)))

def one_sentence_to_ids(sentence, sentence_length=SENTENCE_LENGTH):
    """Convert one sentence to a list of word IDs."

Crop or pad to 0 the sentences to ensure equal length if necessary.
Words without ID are assigned ID 1.
>>> one_sentence_to_ids(['my','first','sentence'], 2)
([11095, 121], 2)
>>> one_sentence_to_ids(['my','ssecond','sentence'], 2)
([11095, 1], 2)
>>> one_sentence_to_ids(['yes'], 2)
([21402, 0], 1)
"""
    vectordb = sqlite3.connect(DB)
    c = vectordb.cursor()
    word_ids = []
    for w in sentence:
        if len(word_ids) >= sentence_length:
            break
        c.execute("""SELECT word_index, word
                  FROM vectors
                  INDEXED BY word_idx
                  WHERE word=?""", (w, ))
        r = c.fetchall()
        if len(r) > 0:
            word_ids.append(r[0][0])
        else:
            word_ids.append(1)
    # Pad with zeros if necessary
    num_words = len(word_ids)
    if num_words < sentence_length:
        word_ids += [0]*(sentence_length-num_words)
    vectordb.close()
    return word_ids, num_words

def parallel_sentences_to_ids(sentences, sentence_length=SENTENCE_LENGTH):
    """Convert each sentence to a list of word IDs.

Crop or pad to 0 the sentences to ensure equal length if necessary.
Words without ID are assigned ID 1.
>>> parallel_sentences_to_ids([['my','first','sentence'],['my','ssecond','sentence'],['yes']], 2)
(([11095, 121], [11095, 1], [21402, 0]), (2, 2, 1))
"""
    #print("Database available at parallel_sentences_to_ids:", DB)
    with Pool() as pool:
        return tuple(zip(*pool.map(partial(one_sentence_to_ids,
                                           sentence_length=sentence_length),
                                   sentences)))

#    vectordb = sqlite3.connect(DB)
#    c = vectordb.cursor()
#    result = []
#    sequence_lengths = []
#    for sentence in sentences:
#        word_ids = []
#        for w in sentence:
#            if len(word_ids) >= sentence_length:
#                break
#            c.execute("""SELECT word_index, word
#                         FROM vectors
#                         INDEXED BY word_idx
#                         WHERE word=?""", (w, ))
#            r = c.fetchall()
#            if len(r) > 0:
#                word_ids.append(r[0][0])
#            else:
#                word_ids.append(1)
#        # Pad with zeros if necessary
#        sequence_lengths.append(len(word_ids))
#        if len(word_ids) < sentence_length:
#            word_ids += [0]*(sentence_length-len(word_ids))
#        result.append(word_ids)
#    vectordb.close()
#    return result, sequence_lengths

def snippets_to_ids(snippets, sentence_length, max_num_snippets=MAX_NUM_SNIPPETS):
    """Convert the snippets to lists of word IDs.
    >>> snippets_to_ids([['sentence', 'one'], ['sentence'], ['two']], 3, 2)
    (([12205, 68, 0], [12205, 0, 0]), (2, 1))
    >>> snippets_to_ids([['sentence', 'three']], 3, 2)
    (([12205, 98, 0], [0, 0, 0]), (2, 0))
    """
    # Pad to the maximum number of snippets
    working_sample = snippets[:max_num_snippets]
    #print("Number of snips: %i" % len(sample))
    if len(working_sample) < max_num_snippets:
        working_sample += [[]] * (max_num_snippets-len(working_sample))

    # Convert to word IDs
    return sentences_to_ids(working_sample, sentence_length)

#def parallel_snippets_to_ids(batch_snippets,
#                             sentence_length,
#                             max_num_snippets=MAX_NUM_SNIPPETS):
#    result = [snippets_to_ids(n, sentence_length, max_num_snippets)
#              for n in batch_snippets]
#    return tuple(zip(*result))

def parallel_snippets_to_ids(batch_snippets,
                             sentence_length,
                             max_num_snippets=MAX_NUM_SNIPPETS):
    """Convert the batch of snippets to lists of word IDs.
    >>> parallel_snippets_to_ids([[['sentence', 'one'], ['sentence'], ['two']],[['sentence', 'three']]], 3, 2)
    ((([12205, 68, 0], [12205, 0, 0]), ([12205, 98, 0], [0, 0, 0])), ((2, 1), (2, 0)))
    """
    with Pool() as pool:
        return tuple(zip(*pool.map(partial(snippets_to_ids,
                                           sentence_length=sentence_length,
                                           max_num_snippets=max_num_snippets),
                                   batch_snippets)))

# def last_relevant_state(output, length):
#     """Return the state at location length-1
# length is the list of lengths of the current batch
# Based on https://danijar.com/variable-sequence-lengths-in-tensorflow/"""
#     batch_size = tf.shape(output)[0]
#     max_length = int(output.get_shape()[1])
#     out_size = int(output.get_shape()[2])
#     index = tf.range(0, batch_size) * max_length + (length - 1)
#     flat = tf.reshape(output, [-1, out_size])
#     relevant = tf.gather(flat, index)
#     return relevant

def embeddings_one_sentence(row):
    return [float(x) for x in row[1].split()]

class BasicNN:
    """A simple NN regressor"""
    def __init__(self, sentence_length=SENTENCE_LENGTH, batch_size=128, embeddings=True,
                 hidden_layer=0, build_graph=False):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.embeddings = None
        self.hidden_layer = hidden_layer
        self.graph = build_graph
        if embeddings:
            vectordb = sqlite3.connect(DB)
            print("Database %s opened" % DB)
            c = vectordb.cursor()
            c_iterator = c.execute("""SELECT word_index, data
                                      FROM vectors""")
            print("Loading word embeddings")
            with Pool() as pool:
                self.embeddings = pool.map(embeddings_one_sentence,
                                           c_iterator)
            print("Word embeddings loaded")
            # self.embeddings = []
            # print("Loading word embeddings")
            # for row in c_iterator:
            #     self.embeddings.append([float(x) for x in row[1].split()])
            # print("Word embeddings loaded")
            vectordb.close()

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "BasicNN%s%s" % (str_embeddings, str_hidden)


    def __build_graph__(self, use_peepholes=False, learningrate=0.001):
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0))
            else:
                embedding_matrix = tf.constant(self.embeddings)

            # Reduce embeddings
            inputs = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
            with tf.name_scope('mean'):
                output_predropout = tf.reduce_mean(inputs, 1)
#                output_predropout = tf.div(tf.reduce_sum(inputs, 1),
#                                           tf.to_float(self.sequence_lengths))
            output = tf.nn.dropout(output_predropout, self.keep_prob)

            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = EMBEDDINGS
                hidden_output = output
            else:
                with tf.name_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([EMBEDDINGS,
                                                                self.hidden_layer],
                                                                stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output = tf.nn.relu(tf.matmul(output,
                                                         W_hidden) + b_hidden)

            # Dropout
            dropout_layer = tf.nn.dropout(hidden_output, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
            with tf.name_scope('linear'):
                self.Y_output = tf.matmul(dropout_layer, W_out) + b_out

            # Optimisation
            with tf.name_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph

    def restore(self, savepath):
        graph = self.__build_graph__()
        self.sess = tf.Session(graph=graph)
        self.saver.restore(self.sess, savepath)
        print("Model restored from file: %s" % savepath)

    def fit(self, X_train, _Q_train, Y_train,
            verbose=2, nb_epoch=3, use_peepholes=False,
            learningrate=0.001, dropoutrate=0.5, savepath=None,
            restore_model=False):
                
        if restore_model:
            print("Restoring BasicNN model from %s" % savepath)
            self.restore(savepath)
            return self.test(X_train, None, Y_train)

        # Training loop
        print("Extracting sentence IDs")
        X, sequence_lengths = parallel_sentences_to_ids(X_train, self.sentence_length)
        print("Sentence IDs extracted")
        return self.__fit__(X, sequence_lengths, None, None, Y_train, 
                            verbose, nb_epoch, use_peepholes, learningrate, 
                            dropoutrate, savepath)
    
    def __fit__(self, X, sequence_lengths, _Q, _sequence_lengths_q, Y, 
                verbose, nb_epoch, use_peepholes, learningrate, dropoutrate,
                savepath=None):

        graph = self.__build_graph__(use_peepholes, learningrate)
        
        self.sess = tf.Session(graph=graph)
        if self.graph:
            for f in glob.glob('logs/*'):
                os.remove(f)
            summary_writer = tf.summary.FileWriter('logs', graph=graph)

        self.sess.run(self.init)

        X = np.array(X)
        sequence_lengths = np.array(sequence_lengths)
        
        random.seed(1234)
        indices = list(range(len(X)))
        for step in range(nb_epoch):
            random.shuffle(indices)
            lastbatch = 0
            batches = list(range(0, len(X), self.batch_size))
            if batches[-1] != len(X):
                # Add last batch
                batches.append(len(X))
            allloss = []
            for bi, batch in enumerate(batches):
                # batch = batches[bi]
                if batch == 0:
                    continue
                feed_dict = {self.X_input: X[indices[lastbatch:batch]],
                             self.Y_input: Y[indices[lastbatch:batch]],
                             self.sequence_lengths: sequence_lengths[indices[lastbatch:batch]],
                             self.keep_prob: dropoutrate}
                self.sess.run(self.train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 1:
                    centile = bi*100/len(batches)
                    bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%3i %s %i%%  Batch Loss %f Mean Loss %f\r" % \
                          (step, bar, centile, thisloss, np.mean(allloss)), end='')
                    sys.stdout.flush()
                lastbatch = batch

            feed_dict = {self.X_input: X[indices[batches[-2]:batches[-1]]],
                         self.Y_input: Y[indices[batches[-2]:batches[-1]]],
                         self.sequence_lengths: sequence_lengths[indices[batches[-2]:batches[-1]]],

                         self.keep_prob: 1}
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            trainloss = self.__test__(X, sequence_lengths, None, None, Y)
            if verbose > 0:
                bar = "["+"="*10+"]"
                print("%3i %s 100%% Batch Loss %f Mean Loss %f Train Loss %f" % \
                      (step, bar, currentloss, meanloss, trainloss))
        if self.graph:
            summary_writer.close()
            
        if savepath:
            save_path = self.saver.save(self.sess, savepath)
            print("Model saved in file: %s" % save_path)
            
        return trainloss

    def predict(self, X_topredict, _Q_topredict):
        X, sequence_lengths = parallel_sentences_to_ids(X_topredict, self.sentence_length)
        feed_dict={self.X_input: X,
                   self.sequence_lengths: sequence_lengths,
                   self.keep_prob: 1}
        return self.sess.run(self.Y_output, feed_dict=feed_dict)

    def __test__(self, X, sequence_lengths, _Q, sequence_lengths_q, Y):
        batches = list(range(0, len(X), self.batch_size))
        if batches[-1] != len(X):
            # Add last batch
            batches.append(len(X))
        #alllosses = []
        allpredictions = []
        lastbatch = 0
        for batch in batches:
            # batch = batches[bi]
            if batch == 0:
                continue
            feed_dict = {self.X_input: X[lastbatch:batch],
                         self.Y_input: Y[lastbatch:batch],
                         self.sequence_lengths: sequence_lengths[lastbatch:batch],
                         self.keep_prob: 1}
            #alllosses.append(self.sess.run(self.loss, feed_dict=feed_dict))
            if allpredictions == []:
                allpredictions = self.sess.run(self.Y_output, feed_dict=feed_dict)
            else:
                allpredictions = np.vstack((allpredictions,
                                            self.sess.run(self.Y_output, feed_dict=feed_dict)))
            lastbatch = batch
        #print()
        #print(len(allpredictions), len(X), len(Y))
        #print(np.mean(alllosses))
        #print(mean_squared_error(Y, allpredictions))
        #print(mean_squared_error(np.ravel(Y), np.ravel(allpredictions)))
        #return np.mean(alllosses)
        return mean_squared_error(Y, allpredictions)

    def test(self, X_test, _Q_test, Y):
        X, sequence_lengths = parallel_sentences_to_ids(X_test, self.sentence_length)
        return self.__test__(X, sequence_lengths, None, None, Y)

class CNN(BasicNN):
    """A convolutional NN regressor"""
    NGRAMS = (2, 3, 4)
    CONVOLUTION = 32

    def __build_graph__(self, use_peepholes=False, learningrate=0.001):
        "Build the CNN graph"
        print("Building CNN graph")
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0))
#                embedding_matrix = tf.zeros([VOCABULARY, EMBEDDINGS])
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # CNN
            with tf.name_scope('CNN'):
                pooled_outputs = []
                for ngram in self.NGRAMS:
                    ngram_filter = tf.Variable(tf.truncated_normal([1,
                                                                    ngram,
                                                                    EMBEDDINGS,
                                                                    self.CONVOLUTION],
                                                                   stddev=0.1))
                    ngram_bias = tf.Variable(tf.constant(0.1,shape=[self.CONVOLUTION]))
                    H_conv = tf.nn.relu(tf.nn.conv2d([tf.nn.embedding_lookup(embedding_matrix,
                                                                            self.X_input)],
                                                     ngram_filter,
                                                     strides=[1,1,1,1],
                                                     padding='VALID') + ngram_bias)
                    H_pool = tf.nn.max_pool(H_conv,
                                            ksize=[1,1,(SENTENCE_LENGTH-ngram+1),1],
                                            strides=[1,1,1,1], padding='VALID')
                    pooled_outputs.append(H_pool)
                pool_layer_size = self.CONVOLUTION*len(self.NGRAMS)
                reshaped_pool = tf.reshape(tf.concat(pooled_outputs, 3),
                                           [-1, pool_layer_size])
            H_concat_drop = tf.nn.dropout(reshaped_pool, self.keep_prob)

            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = pool_layer_size
                hidden_output = H_concat_drop
            else:
                with tf.name_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([pool_layer_size,
                                                                self.hidden_layer],
                                                               stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output_predrop = tf.nn.relu(tf.matmul(H_concat_drop, W_hidden) + b_hidden)
                hidden_output = tf.nn.dropout(hidden_output_predrop, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
            with tf.name_scope('linear'):
                self.Y_output = tf.matmul(hidden_output, W_out) + b_out

            # Optimisation
            with tf.name_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
        return graph

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "CNN%s%s" % (str_embeddings, str_hidden)


class LSTM(BasicNN):
    """A simple LSTM regressor"""
    def __build_graph__(self, use_peepholes=False, learningrate=0.001):
        "Build the graph of the deep learning model"
        print("Building the LSTM graph")
        LSTM_HIDDEN_UNITS = EMBEDDINGS
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0))
#                embedding_matrix = tf.zeros([VOCABULARY, EMBEDDINGS])
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # LSTM
            with tf.name_scope('LSTM'):
                lstm = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                               state_is_tuple=True,
                                               use_peepholes=use_peepholes)
                with tf.name_scope('LSTM_dropout'):
                    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,
                                                                 input_keep_prob=self.keep_prob,
                                                                 output_keep_prob=self.keep_prob)
                inputs = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
                output, state = tf.nn.dynamic_rnn(lstm_dropout, inputs,
                                                  sequence_length=self.sequence_lengths,
                                                  dtype=tf.float32)
            # Extract the last output; based on http://stackoverflow.com/questions/36817596/get-last-output-of-dynamic-rnn-in-tensorflow
#            last_index = tf.shape(output)[1]-1
#            output_rs = tf.transpose(output,[1,0,2])
#            last_state = tf.nn.embedding_lookup(output_rs,[self.sequence_lengths-1])

            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = LSTM_HIDDEN_UNITS
                hidden_output = state.h
            else:
                with tf.name_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([LSTM_HIDDEN_UNITS,
                                                                self.hidden_layer],
                                                                stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output_predrop = tf.nn.relu(tf.matmul(state.h, W_hidden) + b_hidden)
                hidden_output = tf.nn.dropout(hidden_output_predrop, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
#            self.Y_output = tf.matmul(output[:, self.sentence_length-1, :],
            with tf.name_scope('linear'):
                self.Y_output = tf.matmul(hidden_output, W_out) + b_out

            # Optimisation
            with tf.name_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        print("LSTM graph built")
        return graph

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "LSTM%s%s" % (str_embeddings, str_hidden)


# class LSTMEmbeddings(LSTM):
#     """A LSTM regressor that uses fixed sentence embeddings"""
#     def __init__(self, sentence_length=SENTENCE_LENGTH, batch_size=128):
#         LSTM.__init__(self, sentence_length, batch_size)
#         vectordb = sqlite3.connect(DB)
#         c = vectordb.cursor()
#         c_iterator = c.execute("""SELECT word_index, data
#                                   FROM vectors""")
#         self.embeddings = []
#         print("Loading word embeddings")
#         for row in c_iterator:
#             self.embeddings.append([float(x) for x in row[1].split()])
#         print("Word embeddings loaded")
#         vectordb.close()

#     def name(self):
#         return "LSTMEmbeddings"

#     def __build_graph__(self, use_peepholes=False,
#             #embeddings_lambda=10,
#             learningrate=0.001, dropoutrate=0.5):
#         LSTM_HIDDEN_UNITS = EMBEDDINGS
#         graph = tf.Graph()
#         with graph.as_default():
#             self.X_input = tf.placeholder(tf.int32,
#                                           shape=(None,
#                                                  self.sentence_length))
#             self.Y_input = tf.placeholder(tf.float32,
#                                           shape=(None, 1))
#             self.keep_prob = tf.placeholder(tf.float32)
#             self.sequence_lengths = tf.placeholder(tf.int32,
#                                                    shape=(None,))

#             # Embedding
#             embedding_matrix = tf.constant(self.embeddings)
# #            embedding_matrix = tf.Variable(self.embeddings)

#             # LSTM
#             lstm = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
#                                            state_is_tuple=True,
#                                            use_peepholes=use_peepholes)
#             inputs = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
#             output, state = tf.nn.dynamic_rnn(lstm, inputs,
#                                               sequence_length=self.sequence_lengths,
#                                               dtype=tf.float32)

#             # Final prediction
#             W_out = tf.Variable(tf.truncated_normal([LSTM_HIDDEN_UNITS, 1],
#                                                     stddev = 0.1))
#             b_out = tf.Variable(tf.constant(0.1, shape=[1]))
# #            self.Y_output = tf.matmul(output[:,self.sentence_length-1,:],
#             self.Y_output = tf.matmul(state.h,
#                                       W_out) + b_out

#             # Optimisation
#             self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
#             optimizer = tf.train.AdamOptimizer(learningrate)
#             self.train = optimizer.minimize(self.loss)

#             self.init = tf.global_variables_initializer()
#         return graph

class LSTMBidirectional(BasicNN):
    """A bidirectional LSTM regressor"""

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "Bi-LSTM%s%s" % (str_embeddings, str_hidden)

    def __build_graph__(self, use_peepholes=False,
            #embeddings_lambda=10,
            learningrate=0.001):
        LSTM_HIDDEN_UNITS = EMBEDDINGS
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int64,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0)
                )
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # LSTM
            lstm_fw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                              state_is_tuple=True,
                                              use_peepholes=use_peepholes)
#            lstm_fw_dropout = lstm_fw
            lstm_fw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_fw,
                                                            input_keep_prob=self.keep_prob,
                                                            output_keep_prob=self.keep_prob)

            lstm_bw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                              state_is_tuple=True,
                                              use_peepholes=use_peepholes)
#            lstm_bw_dropout = lstm_bw
            lstm_bw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_bw,
                                                            input_keep_prob=self.keep_prob,
                                                            output_keep_prob=self.keep_prob)
            inputs = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
            output, state = tf.nn.bidirectional_dynamic_rnn(
                                              lstm_fw_dropout,
                                              lstm_bw_dropout,
                                              inputs,
                                              sequence_length=self.sequence_lengths,
                                              dtype=tf.float32)
            state_fw, state_bw = state
            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = LSTM_HIDDEN_UNITS*2
                hidden_output = tf.concat([state_fw.h, state_bw.h], 1)
            else:
                with tf.name_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([LSTM_HIDDEN_UNITS*2,
                                                                self.hidden_layer],
                                                                stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output_predrop = tf.nn.relu(tf.matmul(tf.concat([state_fw.h, state_bw.h], 1),
                                                             W_hidden) + b_hidden)
                hidden_output = tf.nn.dropout(hidden_output_predrop, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev = 0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
#            self.Y_output = tf.matmul(output[:,self.sentence_length-1,:],
            with tf.name_scope('linear'):
                self.Y_output = tf.matmul(hidden_output, W_out) + b_out

            # Optimisation
            with tf.name_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph

class Similarities(BasicNN):
    """A regressor that incorporates similarity operations"""
    def __init__(self, sentence_length=SENTENCE_LENGTH, batch_size=128, embeddings=True,
                 hidden_layer=0, build_graph=False, comparison=compare.SimMul(), positions=False):
        BasicNN.__init__(self, sentence_length, batch_size, embeddings,
                 hidden_layer, build_graph)
        self.comparison = comparison
        self.positions = positions
                 
    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "Mean%s%s%s%s" % (self.comparison.name, str_embeddings, str_positions, str_hidden)


    def __build_graph__(self, use_peepholes=False,
            #embeddings_lambda=10,
            learningrate=0.001):
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.X2_input = tf.placeholder(tf.float32,
                                           shape=(None, 1))
            if self.positions:
                positions = [self.X2_input]
            else:
                positions = []
            self.Q_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))
            self.question_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0)
                )
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # Reduce embeddings
            inputs_s = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
            with tf.name_scope('mean_s'):
                output_s_predrop = tf.reduce_mean(inputs_s, 1)
#                output_s_predrop = tf.divide(tf.reduce_sum(inputs_s, 1),
#                                             tf.reshape(tf.to_float(self.sequence_lengths), (-1, 1)))
            output_s = tf.nn.dropout(output_s_predrop, self.keep_prob)
            inputs_q = tf.nn.embedding_lookup(embedding_matrix, self.Q_input)
            with tf.name_scope('mean_q'):
                output_q_predrop = tf.reduce_mean(inputs_q, 1)
#                output_q_predrop = tf.divide(tf.reduce_sum(inputs_q, 1),
#                                             tf.reshape(tf.to_float(self.question_lengths), (-1, 1)))
            output_q = tf.nn.dropout(output_q_predrop, self.keep_prob)

            # Similarity operation
#            W_sim = tf.Variable(tf.truncated_normal([EMBEDDINGS*2],
#                                                    mean = 1.0,
#                                                    stddev = 0.1))
            sim = self.comparison.compare(output_s, output_q, EMBEDDINGS)

            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = EMBEDDINGS + self.comparison.size + len(positions)
                hidden_output = tf.concat([output_s, sim] + positions, 1)
            else:
                with tf.name_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([EMBEDDINGS + self.comparison.size + len(positions),
                                                                self.hidden_layer],
                                                                stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output = tf.nn.relu(tf.matmul(tf.concat([output_s, sim] + positions, 1),
                                                                   W_hidden) + b_hidden)
            # Dropout
            dropout_layer = tf.nn.dropout(hidden_output, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev = 0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
            with tf.name_scope('linear'):
                self.Y_output = tf.matmul(dropout_layer, W_out) + b_out
                #self.Y_output = tf.sigmoid(tf.matmul(dropout_layer, W_out) + b_out)

            # Optimisation
            with tf.name_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input))
                #self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input) * self.Y_input**2)
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            verbose=2, nb_epoch=3, use_peepholes=False,
            learningrate=0.001, dropoutrate=0.5, savepath=None,
            restore_model=False):

        if restore_model:
            print("Restoring Similarities model from %s" % savepath)
            self.restore(savepath)
            return self.test(X_train, Q_train, Y_train, X_positions=X_positions)

        assert(len(X_train) == len(Q_train))
        assert(len(X_train) == len(Y_train))
        
        # Training loop
        X, sequence_lengths_x = parallel_sentences_to_ids(X_train, self.sentence_length)
        sequence_lengths_x = np.array(sequence_lengths_x)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_train, self.sentence_length)
        sequence_lengths_q = np.array(sequence_lengths_q)

        return self.__fit__(X, sequence_lengths_x, X_positions, Q, sequence_lengths_q, Y_train,
                            verbose, nb_epoch, use_peepholes, learningrate, 
                            dropoutrate, savepath)
                            
    def __fit__(self, X, sequence_lengths_x, X_positions, Q, sequence_lengths_q, Y,
                verbose, nb_epoch, use_peepholes, learningrate, dropoutrate,
                savepath=None):
        graph = self.__build_graph__(use_peepholes, learningrate)

        self.sess = tf.Session(graph=graph)
        if self.graph:
            for f in glob.glob('logs/*'):
                os.remove(f)
            summary_writer = tf.summary.FileWriter('logs', graph=graph)

        self.sess.run(self.init)

        X = np.array(X)
        Q = np.array(Q)
        sequence_lengths_x = np.array(sequence_lengths_x)
        sequence_lengths_q = np.array(sequence_lengths_q)

        random.seed(1234)
        indices = list(range(len(X)))
        for step in range(nb_epoch):
            random.shuffle(indices)
            lastbatch = 0
            batches = list(range(0, len(X), self.batch_size))
            if batches[-1] != len(X):
                # Add last batch
                batches.append(len(X))
            allloss = []
            for bi, batch in enumerate(batches):
                # batch = batches[bi]
                if batch == 0:
                    continue
                feed_dict = {self.X_input: X[indices[lastbatch:batch]],
                             self.X2_input: X_positions[indices[lastbatch:batch]],
                             self.Q_input: Q[indices[lastbatch:batch]],
                             self.Y_input: Y[indices[lastbatch:batch]],
                             self.sequence_lengths: sequence_lengths_x[indices[lastbatch:batch]],
                             self.question_lengths: sequence_lengths_q[indices[lastbatch:batch]],
                             self.keep_prob: dropoutrate}
                self.sess.run(self.train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 1:
                    centile = bi*100/len(batches)
                    bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%3i %s %i%%  Batch Loss %f Mean Loss %f\r" % \
                          (step, bar, centile, thisloss, np.mean(allloss)), end='')
                    sys.stdout.flush()
                lastbatch = batch

            feed_dict = {self.X_input: X[indices[batches[-2]:batches[-1]]],
                         self.X2_input: X_positions[indices[batches[-2]:batches[-1]]],
                         self.Q_input: Q[indices[batches[-2]:batches[-1]]],
                         self.Y_input: Y[indices[batches[-2]:batches[-1]]],
                         self.sequence_lengths: sequence_lengths_x[indices[batches[-2]:batches[-1]]],
                         self.question_lengths: sequence_lengths_q[indices[batches[-2]:batches[-1]]],
                         self.keep_prob: 1}
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            trainloss = self.__test__(X, sequence_lengths_x,
                                      Q, sequence_lengths_q,
                                      Y,
                                      X_positions)
            if verbose > 0:
                bar = "["+"="*10+"]"
                print("%3i %s 100%% Batch Loss %f Mean Loss %f Train Loss %f" % \
                      (step, bar, currentloss, meanloss, trainloss))
        if self.graph:
            summary_writer.close()

        if savepath:
            print("Saving model in %s" % savepath)
            save_path = self.saver.save(self.sess, savepath)
            print("Model saved in file: %s" % save_path)

        return trainloss

    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        assert(len(X_topredict) == len(Q_topredict))
        if len(X_topredict) == 0:
            print("WARNING: Data to predict is an empty list")
            return []
        X, sequence_lengths_x = parallel_sentences_to_ids(X_topredict, self.sentence_length)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_topredict, self.sentence_length)
        feed_dict={self.X_input: X,
                   self.X2_input: X_positions,
                   self.sequence_lengths: sequence_lengths_x,
                   self.Q_input: Q,
                   self.question_lengths: sequence_lengths_q,
                   self.keep_prob: 1}
        return self.sess.run(self.Y_output, feed_dict=feed_dict)

    def test(self, X_test, Q_test, Y, X_positions=[]):
        assert(len(X_test) == len(Q_test))
        assert(len(X_test) == len(Y))

        X, sequence_lengths_x = parallel_sentences_to_ids(X_test, self.sentence_length)
        Q, sequence_lengths_q = parallel_sentences_to_ids(Q_test, self.sentence_length)
        return self.__test__(X, sequence_lengths_x,
                             Q, sequence_lengths_q,
                             Y,
                             X_positions)

    def __test__(self, X, sequence_lengths_x,
                       Q, sequence_lengths_q,
                       Y,
                       X_positions):
        batches = list(range(0, len(X), self.batch_size))
        if batches[-1] != len(X):
            # Add last batch
            batches.append(len(X))
        #alllosses = []
        allpredictions = []
        lastbatch = 0
        for batch in batches:
            # batch = batches[bi]
            if batch == 0:
                continue
            feed_dict = {self.X_input: X[lastbatch:batch],
                         self.X2_input: X_positions[lastbatch:batch],
                         self.Q_input: Q[lastbatch:batch],
                         self.Y_input: Y[lastbatch:batch],
                         self.sequence_lengths: sequence_lengths_x[lastbatch:batch],
                         self.question_lengths: sequence_lengths_q[lastbatch:batch],
                         self.keep_prob: 1}
            #alllosses.append(self.sess.run(self.loss, feed_dict=feed_dict))
            if allpredictions == []:
                allpredictions = self.sess.run(self.Y_output, feed_dict=feed_dict)
            else:
                allpredictions = np.vstack((allpredictions,
                                            self.sess.run(self.Y_output, feed_dict=feed_dict)))
            lastbatch = batch
        #return np.mean(alllosses)
        return mean_squared_error(Y, allpredictions)

class Similarities2(BasicNN):
    """A regressor that incorporates similarity operations"""
    def name(self):
        if self.embeddings == None:
            return "Similarities version 2"
        else:
            return "Similarities(embed%i) version 2" % EMBEDDINGS

    def __build_graph__(self, use_peepholes=False,
            #embeddings_lambda=10,
            learningrate=0.001):
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Q_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.SN_input = tf.placeholder(tf.int32,
                                           shape=(None,
                                                  MAX_NUM_SNIPPETS,
                                                  self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))
            self.question_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))
            self.snippets_lengths = tf.placeholder(tf.int32,
                                                shape=(None,
                                                       MAX_NUM_SNIPPETS))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0)
                )
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # Reduce embeddings
            inputs_s = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
            output_s = tf.reduce_mean(inputs_s, 1) # shape: (batchs, EMBEDDINGS)
            inputs_q = tf.nn.embedding_lookup(embedding_matrix, self.Q_input)
            output_q = tf.reduce_mean(inputs_q, 1)

            # Similarity operation
#            W_sim = tf.Variable(tf.truncated_normal([EMBEDDINGS*2],
#                                                    mean = 1.0,
#                                                    stddev = 0.1))
            sim = tf.multiply(output_s, output_q)

            # Max similarity to snippets
            inputs_snips = tf.nn.embedding_lookup(embedding_matrix, self.SN_input)
            output_snips = tf.reduce_mean(inputs_snips, 2) # shape: (batchs, SNIP, EMBEDDINGS)
            mul_snips = tf.multiply(tf.reshape(output_s, [-1, 1, EMBEDDINGS]), output_snips)
            W_mul = tf.Variable(tf.truncated_normal([EMBEDDINGS, 1],
                                                    mean=1.0,
                                                    stddev=0.1))
            sim_snips = tf.matmul(tf.reshape(mul_snips, [-1, EMBEDDINGS]), W_mul)
            max_snips = tf.reduce_max(tf.reshape(sim_snips, [-1, MAX_NUM_SNIPPETS]),
                                      reduction_indices=1,
                                      keep_dims=True)

            ## Hidden layer
            #HIDDEN = 50
            #W_hidden = tf.Variable(tf.truncated_normal([EMBEDDINGS*2+1, HIDDEN],
            #                                           stddev=0.1))
            #b_hidden = tf.Variable(tf.constant(0.1, shape=[HIDDEN]))
            #hidden = tf.nn.relu(tf.matmul(tf.concat([output_s, sim, max_snips], 1),
            #                              W_hidden) + b_hidden)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([EMBEDDINGS*2+1, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
#            self.Y_output = tf.matmul(output[:,self.sentence_length-1,:],
            self.Y_output = tf.matmul(tf.concat([output_s, sim, max_snips], 1),
                                      W_out) + b_out

            # Optimisation
            self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            return graph

    def restore(self, savepath):
        graph = self.__build_graph__()
        self.sess = tf.Session(graph=graph)
        self.saver.restore(self.sess, savepath)
        print("Model restored from file: %s" % savepath)

    def fit(self, X_train, Q_train, Snippets_train, Y_train,
            verbose=2, nb_epoch=3, use_peepholes=False,
            learningrate=0.001, dropoutrate=0.5, savepath=None,
            restore_model=False):

        if restore_model:
            print("Restoring Similarities2 model from %s" % savepath)
            self.restore(savepath)
            return self.test(X_train, None, Y_train)

        assert(len(X_train) == len(Q_train))
        assert(len(X_train) == len(Y_train))

        graph = self.__build_graph__(use_peepholes, learningrate)

        self.sess = tf.Session(graph=graph)
        self.sess.run(self.init)

        # Training loop
        X, sequence_lengths_x = sentences_to_ids(X_train, self.sentence_length)
        X = np.array(X)
        sequence_lengths_x = np.array(sequence_lengths_x)
        Q, sequence_lengths_q = sentences_to_ids(Q_train, self.sentence_length)
        Q = np.array(Q)
        sequence_lengths_q = np.array(sequence_lengths_q)
        SN, sequence_lengths_sn = parallel_snippets_to_ids(Snippets_train, self.sentence_length)
        SN = np.array(SN)
        sequence_lengths_sn = np.array(sequence_lengths_sn)
        #print("Shape of snippets array:", SN.shape)
        #print(SN[0])

        random.seed(1234)
        indices = list(range(len(X)))
        for step in range(nb_epoch):
            random.shuffle(indices)
            lastbatch = 0
            batches = list(range(0, len(X), self.batch_size))
            if batches[-1] != len(X)-1:
                # Add last batch
                batches.append(len(X)-1)
            allloss = []
            for bi, batch in enumerate(batches):
                # batch = batches[bi]
                if batch == 0:
                    continue
                feed_dict = {self.X_input: X[indices[lastbatch:batch]],
                             self.Q_input: Q[indices[lastbatch:batch]],
                             self.SN_input: SN[indices[lastbatch:batch], :],
                             self.Y_input: Y_train[indices[lastbatch:batch]],
                             self.sequence_lengths: sequence_lengths_x[indices[lastbatch:batch]],
                             self.question_lengths: sequence_lengths_q[indices[lastbatch:batch]],
                             self.snippets_lengths: sequence_lengths_sn[indices[lastbatch:batch]],
                             self.keep_prob: dropoutrate}
                self.sess.run(self.train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 1:
                    centile = bi*100/len(batches)
                    bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%3i %s %i%%  Batch Loss %f Mean Loss %f\r" % \
                          (step, bar, centile, thisloss, np.mean(allloss)), end='')
                    sys.stdout.flush()
                lastbatch = batch

            feed_dict = {self.X_input: X[indices[batches[-2]:batches[-1]]],
                         self.Q_input: Q[indices[batches[-2]:batches[-1]]],
                         self.SN_input: SN[indices[batches[-2]:batches[-1]], :],
                         self.Y_input: Y_train[indices[batches[-2]:batches[-1]]],
                         self.sequence_lengths: sequence_lengths_x[indices[batches[-2]:batches[-1]]],
                         self.question_lengths: sequence_lengths_q[indices[batches[-2]:batches[-1]]],
                         self.snippets_lengths: sequence_lengths_sn[indices[batches[-2]:batches[-1]]],
                         self.keep_prob: 1}
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            trainloss = self._test_(X, sequence_lengths_x,
                                    Q, sequence_lengths_q,
                                    SN, sequence_lengths_sn,
                                    Y_train)
            if verbose > 0:
                bar = "["+"="*10+"]"
                print("%3i %s 100%% Batch Loss %f Mean Loss %f Train Loss %f" % \
                      (step, bar, currentloss, meanloss, trainloss))
                      
        if savepath:
            save_path = self.saver.save(self.sess, savepath)
            print("Model saved in file: %s" % save_path)
        
        return trainloss

    def predict(self, X_topredict, Q_topredict, Snippets_topredict):
        assert(len(X_topredict) == len(Q_topredict))
        X, sequence_lengths_x = sentences_to_ids(X_topredict, self.sentence_length)
        Q, sequence_lengths_q = sentences_to_ids(Q_topredict, self.sentence_length)
        SN, sequence_lengths_sn = parallel_snippets_to_ids(Snippets_topredict, self.sentence_length)
        feed_dict={self.X_input: X,
                   self.sequence_lengths: sequence_lengths_x,
                   self.Q_input: Q,
                   self.question_lengths: sequence_lengths_q,
                   self.SN_input: SN,
                   self.snippets_lengths: sequence_lengths_sn,
                   self.keep_prob: 1}
        return self.sess.run(self.Y_output, feed_dict=feed_dict)

    def _test_(self, X, sequence_lengths_x,
                     Q, sequence_lengths_q,
                     SN, sequence_lengths_sn,
                     Y):
        batches = list(range(0, len(X), self.batch_size))
        if batches[-1] != len(X)-1:
            # Add last batch
            batches.append(len(X)-1)
        alllosses = []
        lastbatch = 0
        for batch in batches:
            # batch = batches[bi]
            if batch == 0:
                continue
            feed_dict = {self.X_input: X[lastbatch:batch],
                         self.Q_input: Q[lastbatch:batch],
                         self.Y_input: Y[lastbatch:batch],
                         self.SN_input: SN[lastbatch:batch],
                         self.sequence_lengths: sequence_lengths_x[lastbatch:batch],
                         self.question_lengths: sequence_lengths_q[lastbatch:batch],
                         self.snippets_lengths: sequence_lengths_sn[lastbatch:batch],
                         self.keep_prob: 1}
            alllosses.append(self.sess.run(self.loss, feed_dict=feed_dict))
            lastbatch = batch
        return np.mean(alllosses)


    def test(self, X_test, Q_test, Snippets_test, Y):
        assert(len(X_test) == len(Q_test))
        assert(len(X_test) == len(Y))

        X, sequence_lengths_x = sentences_to_ids(X_test, self.sentence_length)
        Q, sequence_lengths_q = sentences_to_ids(Q_test, self.sentence_length)
        SN, sequence_lengths_sn = parallel_snippets_to_ids(Snippets_test, self.sentence_length)

        return self._test_(X, sequence_lengths_x,
                                    Q, sequence_lengths_q,
                                    SN, sequence_lengths_sn,
                                    Y)

class CNNSimilarities(Similarities):
    """A regressor that incorporates similarity operations"""
    NGRAMS = (2, 3, 4)
    CONVOLUTION = 32

    def name(self):
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "CNN%s%s%s" % (self.comparison.name, str_embeddings, str_hidden)

    def __build_graph__(self, use_peepholes=False,
            #embeddings_lambda=10,
            learningrate=0.001):
        #print("Building CNNSimilarities graph")
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Q_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))
            self.question_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0))
#                embedding_matrix = tf.zeros([VOCABULARY, EMBEDDINGS])
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # CNN for sentences
            pool_layer_size = self.CONVOLUTION*len(self.NGRAMS)
            with tf.name_scope('CNN_s'):
                pooled_outputs = []
                for ngram in self.NGRAMS:
                    ngram_filter = tf.Variable(tf.truncated_normal([1,
                                                                    ngram,
                                                                    EMBEDDINGS,
                                                                    self.CONVOLUTION],
                                                                    stddev=0.1))
                    ngram_bias = tf.Variable(tf.constant(0.1,shape=[self.CONVOLUTION]))
                    H_conv = tf.nn.relu(tf.nn.conv2d([tf.nn.embedding_lookup(embedding_matrix,
                                                                             self.X_input)],
                                                     ngram_filter,
                                                     strides=[1,1,1,1],
                                                     padding='VALID') + ngram_bias)
                    H_pool = tf.nn.max_pool(H_conv,
                                            ksize=[1,1,(SENTENCE_LENGTH-ngram+1),1],
                                            strides=[1,1,1,1], padding='VALID')
                    pooled_outputs.append(H_pool)
                reshaped_pool = tf.reshape(tf.concat(pooled_outputs, 3),
                                           [-1, pool_layer_size])
            H_concat_drop = tf.nn.dropout(reshaped_pool,self.keep_prob)

            # CNN for questions
            with tf.name_scope('CNN_q'):
                pooled_q_outputs = []
                for ngram in self.NGRAMS:
                    ngram_q_filter = tf.Variable(tf.truncated_normal([1,
                                                                      ngram,
                                                                      EMBEDDINGS,
                                                                      self.CONVOLUTION],
                                                                      stddev=0.1))
                    ngram_q_bias = tf.Variable(tf.constant(0.1,shape=[self.CONVOLUTION]))
                    H_q_conv = tf.nn.relu(tf.nn.conv2d([tf.nn.embedding_lookup(embedding_matrix,
                                                                               self.Q_input)],
                                                       ngram_q_filter,
                                                       strides=[1,1,1,1],
                                                       padding='VALID') + ngram_q_bias)
                    H_q_pool = tf.nn.max_pool(H_q_conv,
                                              ksize=[1,1,(SENTENCE_LENGTH-ngram+1),1],
                                              strides=[1,1,1,1], padding='VALID')
                    pooled_q_outputs.append(H_q_pool)
                    reshaped_q_pool = tf.reshape(tf.concat(pooled_q_outputs, 3),
                                                 [-1, pool_layer_size])
            H_concat_drop_q = tf.nn.dropout(reshaped_q_pool,self.keep_prob)

            # Similarity operation
            sim = self.comparison.compare(H_concat_drop, H_concat_drop_q, pool_layer_size)

            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = pool_layer_size + self.comparison.size
                hidden_output = tf.concat([H_concat_drop, sim], 1)
            else:
                with tf.name_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([pool_layer_size + self.comparison.size,
                                                                self.hidden_layer],
                                                                stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output_predrop = tf.nn.relu(tf.matmul(tf.concat([H_concat_drop, sim],
                                                                           1),
                                                                 W_hidden) + b_hidden)
                hidden_output = tf.nn.dropout(hidden_output_predrop, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
#            self.Y_output = tf.matmul(output[:, self.sentence_length-1, :],
            with tf.name_scope('linear'):
                self.Y_output = tf.matmul(hidden_output, W_out) + b_out

            # Optimisation
            with tf.name_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph

class LSTMSimilarities(Similarities):
    """A regressor that incorporates similarity operations"""
    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.embeddings == None:
            str_embeddings = ""
        else:
            str_embeddings = "(embed%i)" % EMBEDDINGS
        if self.hidden_layer == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer
        return "DynBiLSTM%s%s%s%s" % (self.comparison.name, str_embeddings, str_positions, str_hidden)

    def __build_graph__(self, use_peepholes=False,
            #embeddings_lambda=10,
            learningrate=0.001):
        LSTM_HIDDEN_UNITS = EMBEDDINGS
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.X2_input = tf.placeholder(tf.float32,
                                           shape=(None, 1))
            if self.positions:
                positions = [self.X2_input]
            else:
                positions = []
            self.Q_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int64,
                                                   shape=(None,))
            self.question_lengths = tf.placeholder(tf.int64,
                                                   shape=(None,))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0)
                )
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # LSTM for sentences
            with tf.variable_scope("BiLSTM_s"):
                lstm_sfw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
#                lstm_sfw_dropout = lstm_sfw
                lstm_sfw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_sfw,
                                                                 input_keep_prob=self.keep_prob,
                                                                 output_keep_prob=self.keep_prob)
                lstm_sbw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
#                lstm_sbw_dropout = lstm_sbw
                lstm_sbw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_sbw,
                                                                 input_keep_prob=self.keep_prob,
                                                                 output_keep_prob=self.keep_prob)
                inputs_s = tf.nn.embedding_lookup(embedding_matrix, self.X_input)
                output_s, state_s = tf.nn.bidirectional_dynamic_rnn(lstm_sfw_dropout,
                                                                    lstm_sbw_dropout,
                                                                    inputs_s,
                                                                    sequence_length=self.sequence_lengths,
                                                                    dtype=tf.float32)
                state_sfw, state_sbw = state_s

            # LSTM for questions
            with tf.variable_scope("BiLSTM_q"):
                lstm_qfw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
#                lstm_qfw_dropout = lstm_qfw
                lstm_qfw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_qfw,
                                                                 input_keep_prob=self.keep_prob,
                                                                 output_keep_prob=self.keep_prob)
                lstm_qbw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
#                lstm_qbw_dropout = lstm_qbw
                lstm_qbw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_qbw,
                                                                 input_keep_prob=self.keep_prob,
                                                                 output_keep_prob=self.keep_prob)
                inputs_q = tf.nn.embedding_lookup(embedding_matrix, self.Q_input)
                output_q, state_q = tf.nn.bidirectional_dynamic_rnn(lstm_qfw_dropout,
                                                                    lstm_qbw_dropout,
                                                                    inputs_q,
                                                                    sequence_length=self.question_lengths,
                                                                    dtype=tf.float32)
                state_qfw, state_qbw = state_q

            # Similarity operation
#            sim_fw = tf.multiply(state_sfw.h, state_qfw.h)
#            sim_bw = tf.multiply(state_sbw.h, state_qbw.h)
            
            s_concat = tf.concat([state_sfw.h, state_sbw.h], 1)
            q_concat = tf.concat([state_qfw.h, state_qbw.h], 1)
            sim = self.comparison.compare(s_concat, q_concat, LSTM_HIDDEN_UNITS*2)

            # Hidden layer
            if self.hidden_layer == 0:
                hidden_output_size = LSTM_HIDDEN_UNITS*2 + self.comparison.size + len(positions)
                hidden_output = tf.concat([state_sfw.h, state_sbw.h, sim] + positions, 1)
            else:
                with tf.variable_scope('hidden'):
                    W_hidden = tf.Variable(tf.truncated_normal([LSTM_HIDDEN_UNITS*2 + self.comparison.size + len(positions),
                                                                self.hidden_layer],
                                                               stddev=0.1))
                    b_hidden = tf.Variable(tf.constant(0.1, shape=[self.hidden_layer]))
                    hidden_output_size = self.hidden_layer
                    hidden_output_predrop = tf.nn.relu(tf.matmul(tf.concat([state_sfw.h,
                                                                            state_sbw.h,
                                                                            sim] + positions,
                                                                           1),
                                                                 W_hidden) + b_hidden)
                hidden_output = tf.nn.dropout(hidden_output_predrop, self.keep_prob)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([hidden_output_size, 1],
                                                    stddev = 0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
#            self.Y_output = tf.matmul(output[:,self.sentence_length-1,:],
            with tf.variable_scope('linear'):
                self.Y_output = tf.matmul(hidden_output, W_out) + b_out
            # Optimisation
            with tf.variable_scope('MSE'):
                self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        return graph

class LSTMSimilarities2(Similarities2):
    """A regressor that incorporates similarity operations"""
    def name(self):
        if self.embeddings == None:
            return "LSTM Similarities version 2"
        else:
            return "LSTM(embed%i) Similarities version 2" % EMBEDDINGS

    def __build_graph__(self, use_peepholes=False,
            #embeddings_lambda=10,
            learningrate=0.001):
        LSTM_HIDDEN_UNITS = EMBEDDINGS
        graph = tf.Graph()
        with graph.as_default():
            self.X_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.Q_input = tf.placeholder(tf.int32,
                                          shape=(None,
                                                 self.sentence_length))
            self.SN_input = tf.placeholder(tf.int32,
                                           shape=(None,
                                                  MAX_NUM_SNIPPETS,
                                                  self.sentence_length))
            self.Y_input = tf.placeholder(tf.float32,
                                          shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)
            self.sequence_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))
            self.question_lengths = tf.placeholder(tf.int32,
                                                   shape=(None,))
            self.snippets_lengths = tf.placeholder(tf.int32,
                                                shape=(None,
                                                       MAX_NUM_SNIPPETS))

            # Embedding
            if self.embeddings == None:
                embedding_matrix = tf.Variable(
                    tf.random_uniform([VOCABULARY, EMBEDDINGS], -1.0, 1.0)
                )
            else:
                embedding_matrix = tf.constant(self.embeddings)
#                embedding_matrix = tf.Variable(self.embeddings)

            # LSTM for sentences
            with tf.variable_scope("sentences"):
                lstm_sfw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
                lstm_sbw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
                inputs_s = [tf.nn.embedding_lookup(embedding_matrix, self.X_input[:,i]) for i in range(self.sentence_length)]
                output_s, state_sfw, state_sbw = tf.nn.bidirectional_rnn(lstm_sfw,
                                                                         lstm_sbw,
                                                                         inputs_s,
                                                                         sequence_length=self.sequence_lengths,
                                                                         dtype=tf.float32)

            # LSTM for questions
            with tf.variable_scope("questions"):
                lstm_qfw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
                lstm_qbw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
                inputs_q = [tf.nn.embedding_lookup(embedding_matrix, self.Q_input[:,i]) for i in range(self.sentence_length)]
                output_q, state_qfw, state_qbw = tf.nn.bidirectional_rnn(lstm_qfw,
                                                                         lstm_qbw,
                                                                         inputs_q,
                                                                         sequence_length=self.question_lengths,
                                                                         dtype=tf.float32)
            # Similarity operation
            sim = tf.concat([tf.multiply(state_sfw.h, state_qfw.h),
                             tf.multiply(state_sbw.h, state_qbw.h)], 1)

            # Max similarity to snippets
            with tf.variable_scope("snippets"):
                lstm_sfw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
                lstm_sbw = tf.contrib.rnn.LSTMCell(LSTM_HIDDEN_UNITS,
                                                   state_is_tuple=True,
                                                   use_peepholes=use_peepholes)
                inputs_snips = [tf.nn.embedding_lookup(embedding_matrix, self.SN_input[:,:,i]) for i in range(self.sentence_length)]
                state_snips = []
                for i in range(MAX_NUM_SNIPPETS):
                    output_snips, state_snfw, state_snbw = tf.nn.bidirectional_rnn(lstm_sfw,
                                                                                   lstm_sbw,
                                                                                   inputs_snips[:,i,:],
                                                                                   sequence_length=self.snippets_lengths[i],
                                                                                   dtype=tf.float32)
                    state_snips.append(tf.concat([state_snfw, state_snbw], 1))

            mul_snips = tf.multiply(tf.reshape(output_s, [-1, 1, LSTM_HIDDEN_UNITS*2]), state_snips) # TODO: Check this line
            W_mul = tf.Variable(tf.truncated_normal([EMBEDDINGS, 1],
                                                    mean=1.0,
                                                    stddev=0.1))
            sim_snips = tf.matmul(tf.reshape(mul_snips, [-1, EMBEDDINGS]), W_mul)
            max_snips = tf.reduce_max(tf.reshape(sim_snips, [-1, MAX_NUM_SNIPPETS]),
                                      reduction_indices=1,
                                      keep_dims=True)

            # Hidden layer
            HIDDEN = 50
            W_hidden = tf.Variable(tf.truncated_normal([EMBEDDINGS*2+1, HIDDEN],
                                                       stddev=0.1))
            b_hidden = tf.Variable(tf.constant(0.1, shape=[HIDDEN]))
            hidden = tf.nn.relu(tf.matmul(tf.concat([output_s, sim, max_snips], 1),
                                          W_hidden) + b_hidden)

            # Final prediction
            W_out = tf.Variable(tf.truncated_normal([HIDDEN, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
#            self.Y_output = tf.matmul(output[:,self.sentence_length-1,:],
            self.Y_output = tf.matmul(hidden, W_out) + b_out

            # Optimisation
            self.loss = tf.reduce_mean(tf.square(self.Y_output - self.Y_input)) # + embeddings_lambda*tf.reduce_mean(tf.square(self.embeddings-embedding_matrix))
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph

if __name__ == "__main__":
    import doctest
    doctest.testmod()

#    sys.exit()

    import csv
    import re

    cleantext = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split()

    with open('golden_13PubMed.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        sentences = []
        scores = []
        s_ids = []
        for row in reader:
            sentences.append(cleantext(row['sentence text']))
            scores.append([float(row['SU4'])])
            s_ids.append([int(row['sentid'])])

        print("Data has %i items" % len(sentences))
        # print(sentences[:3])
        # print(scores[:3])

        lstm = Similarities(hidden_layer=50, build_graph=True, positions=True)
        print("Training %s" % lstm.name())
        loss = lstm.fit(sentences[100:], sentences[100:], np.array(scores[100:]),
                        X_positions=np.array(s_ids[100:]), nb_epoch=3)
        print("Training loss: %f" % loss)
        testloss = lstm.test(sentences[:100], sentences[:100], scores[:100],
                             X_positions=s_ids[:100])
        print("Test loss: %f" % testloss)
