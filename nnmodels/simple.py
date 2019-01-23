"Simple NN models for regression"
import tensorflow as tf
import numpy as np
import sys
# import time
import random

from nnmodels import compare

class LinearRegression():
    """Simple linear regression"""
    name = "NNR Linear"

    def fit(self, x_train, y_train, verbose=1, learningrate=0.1):
        """Train the regressor
        x_train and y_train must be a numpy array or a CSR matrix
        """
        nb_epoch = 20
        batch_size = 128

        graph = tf.Graph()
        with graph.as_default():
            self.x_input = tf.placeholder(tf.float32,
                                          shape=(None, x_train.shape[1]))
            self.y_input = tf.placeholder(tf.float32, shape=(None, 1))

            if verbose > 0:
                print(self.name)

            # Output layer
            w_matrix = tf.Variable(tf.truncated_normal([x_train.shape[1], 1],
                                                       stddev=0.1))
            b_vector1 = tf.Variable(tf.constant(0.1, shape=[1]))
            self.y_output = tf.matmul(self.x_input, w_matrix)+b_vector1

            # Optimisation
            self.loss = tf.reduce_mean(tf.square(self.y_output - self.y_input))
            optimizer = tf.train.GradientDescentOptimizer(learningrate)
    #        optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=graph)
        self.sess.run(init)

        x_data = x_train#.tocsr()
        lastloss = 0
        random.seed(1234)
        indices = list(range(x_train.shape[0]))
        for step in range(0, nb_epoch):
            random.shuffle(indices)
            lastbatch = 0
            batches = range(0, x_train.shape[0], batch_size)
            if verbose > 0:
                print("Epoch %i/%i" % (step, nb_epoch))
            allloss = []
            for (batch_i, batch) in enumerate(batches):
                #if batch_i > 100:
                #    break
                #batch = batches[batch_i]
                if batch == 0:
                    continue
                feed_dict = {
                    self.x_input: x_data[indices[lastbatch:batch], :].toarray(),
                    self.y_input: y_train[indices[lastbatch:batch]]
                }
                self.sess.run(train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 0:
                    centile = batch_i*100/len(batches)
                    status_bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%s %i%%  Batch Loss %f Mean Loss %f\r"
                          % (status_bar, centile, thisloss, np.mean(allloss)), end='')
                    sys.stdout.flush()
                #time.sleep(0.5)
                lastbatch = batch
            feed_dict = {
                self.x_input: x_data[indices[batches[-2]:batches[-1]], :].toarray(),
                self.y_input: y_train[indices[batches[-2]:batches[-1]]]
            }
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            if verbose > 0:
                status_bar = "["+"="*10+"]"
                print("%s 100%% Batch Loss %f Mean Loss %f"
                      % (status_bar, currentloss, meanloss))
            if verbose > 1:
                feed_dict = {self.x_input: x_data[:10, :].toarray(),
                             self.y_input: y_train[:10]}
                predictions = self.sess.run(self.y_output, feed_dict=feed_dict)
                labels = y_train[:10]
                for i in range(10):
                    print("%i prediction: %1.5f target: %1.5f difference: %1.5f"
                          % (i,
                             predictions[i, 0],
                             labels[i, 0],
                             abs(predictions[i, 0]-labels[i, 0])))

            if abs(meanloss-lastloss) < 0.000001:
                break
            lastloss = meanloss
        return lastloss

    def predict(self, x_data):
        "X must be a numpy array or a CSR matrix"
        return self.sess.run(
            self.y_output,
            feed_dict={self.x_input: np.float32(x_data.toarray())}
        )

    def test(self, x_test, y_test):
        "x_test and Y must be a numpy array or a CSR matrix"
        batch_size = 128
        x_data = x_test#.tocsr()
        batches = range(0, x_data.shape[0], batch_size)
        alllosses = []
        lastbatch = 0
        for batch in batches:
            #batch = batches[batch_i]
            if batch == 0:
                continue
            feed_dict = {self.x_input: x_data[lastbatch:batch, :].toarray(),
                         self.y_input: y_test[lastbatch:batch]}
            alllosses.append(self.sess.run(self.loss, feed_dict))
            lastbatch = batch
        return np.mean(alllosses)

class SingleNNR:
    """NN regressor with one hidden layer"""
    def __init__(self, layer1=50, batch_size=128):
        self.layer1 = layer1
        self.batch_size = batch_size
        self.name = "NNR %i" % (self.layer1)
        
    def __build_graph__(self, dimensions, learningrate=0.001):
        graph = tf.Graph()
        with graph.as_default():

            self.x_input = tf.placeholder(tf.float32,
                                          shape=(None, dimensions))
            self.y_input = tf.placeholder(tf.float32, shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)

            # Hidden layer
    #        w_matrix1 = tf.Variable(tf.zeros([x_train.shape[1], layer1]))
    #        w_matrix1 = tf.Variable(tf.random_uniform([x_train.shape[1], layer1], -1.0, 1.0))
            w_matrix1 = tf.Variable(tf.truncated_normal([dimensions,
                                                         self.layer1],
                                                        stddev=0.1))
            b_vector1 = tf.Variable(tf.constant(0.1, shape=[1]))
            hidden1 = tf.nn.relu(tf.matmul(self.x_input, w_matrix1)+b_vector1)
            hidden1_drop = tf.nn.dropout(hidden1, self.keep_prob)

            # Output layer
            # w_matrix3 = tf.Variable(tf.random_uniform([layer2,1], -1.0, 1.0))
            w_matrix3 = tf.Variable(tf.truncated_normal([self.layer1, 1],
                                                        stddev=0.1))
            b_vector3 = tf.Variable(tf.constant(0.1, shape=[1]))
            self.y_output = tf.matmul(hidden1_drop, w_matrix3)+b_vector3


            self.loss = tf.reduce_mean(tf.square(self.y_output - self.y_input))
    #        optimizer = tf.train.GradientDescentOptimizer(learningrate)
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph

    def restore(self, savepath, dimensions):
        graph = self.__build_graph__(dimensions)
        self.sess = tf.Session(graph=graph)
        self.saver.restore(self.sess, savepath)
        print("Model restored from file: %s" % savepath)


    def fit(self, x_train, y_train, verbose=1, learningrate=0.001, nb_epoch=20,
                  dropoutrate=0.5, savepath=None):
        "x_train and y_train must be a numpy array or a CSR matrix"

        if verbose > 0:
            print("%s Learning rate: %f" % (self.name, learningrate))

        graph = self.__build_graph__(x_train.shape[1], learningrate)

        self.sess = tf.Session(graph=graph)
        self.sess.run(self.init)

        x_data = x_train#.tocsr()
        lastloss = 0
        random.seed(1234)
        indices = list(range(x_train.shape[0]))
        for step in range(0, nb_epoch):
            random.shuffle(indices) # shuffle the data in each epoch
            lastbatch = 0
            batches = list(range(0, x_train.shape[0], self.batch_size))
            if batches[-1] != x_data.shape[0]:
                # Add last batch
                batches.append(x_data.shape[0])
            if verbose > 0:
                print("Epoch %i/%i" % (step, nb_epoch))
            allloss = []
            for batch_i, batch in enumerate(batches):
                #if batch_i > 100:
                #    break
                #batch = batches[batch_i]
                if batch == 0:
                    continue
                if isinstance(x_data, np.ndarray):
                    feed_dict = {
                        self.x_input: x_data[indices[lastbatch:batch], :],
                        self.y_input: y_train[indices[lastbatch:batch]],
                        self.keep_prob: dropoutrate
                        }
                else:                    
                    feed_dict = {
                        self.x_input: x_data[indices[lastbatch:batch], :].todense(),
                        self.y_input: y_train[indices[lastbatch:batch]],
                        self.keep_prob: dropoutrate
                        }
                self.sess.run(self.train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 1:
                    centile = batch_i*100/len(batches)
                    status_bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%s %i%%  Batch Loss %f Mean Loss %f\r"
                          % (status_bar, centile, thisloss, np.mean(allloss)),
                          end='')
                    sys.stdout.flush()
                #time.sleep(0.5)
                lastbatch = batch
            if isinstance(x_data, np.ndarray):
                feed_dict = {
                    self.x_input: x_data[indices[batches[-2]:batches[-1]], :],
                    self.y_input: y_train[indices[batches[-2]:batches[-1]]],
                    self.keep_prob: 1 # Change dropout rate here, e.g. 0.5
                    }
            else:
                feed_dict = {
                    self.x_input: x_data[indices[batches[-2]:batches[-1]], :].todense(),
                    self.y_input: y_train[indices[batches[-2]:batches[-1]]],
                    self.keep_prob: 1 # Change dropout rate here, e.g. 0.5
                    }
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            if verbose > 0:
                status_bar = "["+"="*10+"]"
                print("%s 100%% Batch Loss %f Mean Loss %f" % (status_bar,
                                                               currentloss,
                                                               meanloss))
#            if verbose > 2:
#                feed_dict = {
#                    self.x_input: x_data[:10, :].todense(),
#                    self.y_input: y_train[:10],
#                    self.keep_prob: 1  # Change dropout rate here, e.g. 0.5
#                }
#                predictions = self.sess.run(self.y_output, feed_dict=feed_dict)
#                labels = y_train[:10]
#                for i in range(10):
#                    print(
#                        "%i prediction: %1.5f target: %1.5f difference: %1.5f"
#                        % (i,
#                           predictions[i, 0],
#                           labels[i, 0],
#                           abs(predictions[i, 0]-labels[i, 0]))
#                    )

#            if abs(meanloss-lastloss) < 0.000001:
#                break
            lastloss = meanloss

        if savepath:
            save_path = self.saver.save(self.sess, savepath)
            print("Model saved in file: %s" % save_path)
            
        return lastloss

    def predict(self, x_data):
        "x_data must be a numpy array or a CSR matrix"
        batches = list(range(0, x_data.shape[0], self.batch_size))
        if batches[-1] != x_data.shape[0]:
            batches.append(x_data.shape[0])
        all_y = []
        lastbatch = 0
        for (batch_i, batch) in enumerate(batches):
            batch = batches[batch_i]
            if batch == 0:
                continue
            if isinstance(x_data, np.ndarray):
                feed_dict = {self.x_input: x_data[lastbatch:batch, :],
                             self.keep_prob: 1}
            else:
                feed_dict = {self.x_input: x_data[lastbatch:batch, :].todense(),
                             self.keep_prob: 1}
            all_y.append(self.sess.run(self.y_output, feed_dict))
            lastbatch = batch
        return np.vstack(all_y)
        
#        return self.sess.run(
#            self.y_output,
#            feed_dict={self.x_input: np.float32(x_data),#.todense()),
#                       self.keep_prob:1}
#        )

    def test(self, x_test, y_test):
        "x_test and y_test must be a numpy array or a CSR matrix"
        x_data = x_test#.tocsr()
        batches = list(range(0, x_data.shape[0], self.batch_size))
        if batches[-1] != x_data.shape[0]:
            batches.append(x_data.shape[0])
        alllosses = []
        lastbatch = 0
        for (batch_i, batch) in enumerate(batches):
            batch = batches[batch_i]
            if batch == 0:
                continue
            if isinstance(x_data, np.ndarray):
                feed_dict = {self.x_input: x_data[lastbatch:batch, :],
                             self.y_input: y_test[lastbatch:batch],
                             self.keep_prob: 1}
            else:
                feed_dict = {self.x_input: x_data[lastbatch:batch, :].todense(),
                             self.y_input: y_test[lastbatch:batch],
                             self.keep_prob: 1}
            alllosses.append(self.sess.run(self.loss, feed_dict))
            lastbatch = batch
        return np.mean(alllosses)

class DoubleNNR:
    """NN regressor with two hidden layers"""

    def __init__(self, layer1=50, layer2=50):
        self.layer1 = layer1
        self.layer2 = layer2
        self.name = "NNR %i - %i" % (self.layer1, self.layer2)

    def fit(self, x_train, y_train, verbose=1, learningrate=0.1, nb_epoch=20):
        "x_train and y_train must be a numpy array or a CSR matrix"
        batch_size = 128

        self.x_input = tf.placeholder(tf.float32,
                                      shape=(None, x_train.shape[1]))
        self.y_input = tf.placeholder(tf.float32, shape=(None, 1))
        self.keep_prob = tf.placeholder(tf.float32)

        if verbose > 0:
            print("%s Learning rate: %f" % (self.name, learningrate))

        # Hidden layer
        #w_matrix1 = tf.Variable(tf.zeros([x_train.shape[1],self.layer1]))
        #w_matrix1 = tf.Variable(tf.random_uniform([x_train.shape[1],self.layer1], -1.0, 1.0))
        w_matrix1 = tf.Variable(tf.truncated_normal([x_train.shape[1],
                                                     self.layer1],
                                                    stddev=0.1))
        b_vector1 = tf.Variable(tf.constant(0.1, shape=[1]))
        hidden1 = tf.nn.relu(tf.matmul(self.x_input, w_matrix1)+b_vector1)
        hidden1_drop = tf.nn.dropout(hidden1, self.keep_prob)

        # Hidden layer 2
        #w_matrix2 = tf.Variable(tf.zeros([self.layer1,self.layer2]))
        #w_matrix2 = tf.Variable(tf.random_uniform([self.layer1,self.layer2], -1.0, 1.0))
        w_matrix2 = tf.Variable(tf.truncated_normal([self.layer1, self.layer2],
                                                    stddev=0.1))
        b_vector2 = tf.Variable(tf.constant(0.1, shape=[1]))
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, w_matrix2)+b_vector2)
        hidden2_drop = tf.nn.dropout(hidden2, self.keep_prob)

        # Output layer
        #w_matrix3 = tf.Variable(tf.random_uniform([self.layer2,1], -1.0, 1.0))
        w_matrix3 = tf.Variable(tf.truncated_normal([self.layer2, 1],
                                                    stddev=0.1))
        b_vector3 = tf.Variable(tf.constant(0.1, shape=[1]))
        self.y_output = tf.matmul(hidden2_drop, w_matrix3)+b_vector3

        self.loss = tf.reduce_mean(tf.square(self.y_output - self.y_input))
#        optimizer = tf.train.GradientDescentOptimizer(learningrate)
        optimizer = tf.train.AdamOptimizer(learningrate)
        train = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        x_data = x_train#.tocsr()
        y_data = np.array(y_train)

        # Run the training
        lastloss = 0
        random.seed(1234)
        indices = list(range(x_train.shape[0]))
        for step in range(0, nb_epoch):
            random.shuffle(indices)
            lastbatch = 0
            batches = range(0, x_data.shape[0], batch_size)
            if verbose > 0:
                print("Epoch %i/%i" % (step, nb_epoch))
            allloss = []
            for (batch_i, batch) in enumerate(batches):
                #if batch_i > 100:
                #    break
                #batch = batches[batch_i]
                if batch == 0:
                    continue
                feed_dict = {
                    self.x_input: x_data[indices[lastbatch:batch], :].toarray(),
                    self.y_input: y_data[indices[lastbatch:batch]],
                    self.keep_prob: 0.5  # Change dropout rate here, e.g. 0.5
                }
                self.sess.run(train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 0:
                    centile = batch_i*100/len(batches)
                    status_bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%s %i%%  Batch Loss %f Mean Loss %f\r"
                          % (status_bar, centile, thisloss, np.mean(allloss)),
                          end='')
                    sys.stdout.flush()
                #time.sleep(0.5)
                lastbatch = batch
            feed_dict = {
                self.x_input: x_data[indices[batches[-2]:batches[-1]], :].toarray(),
                self.y_input: y_data[indices[batches[-2]:batches[-1]]],
                self.keep_prob: 1  # Change dropout rate here, e.g. 0.5
            }
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            if verbose > 0:
                status_bar = "["+"="*10+"]"
                print("%s 100%% Batch Loss %f Mean Loss %f"
                      % (status_bar, currentloss, meanloss))
            if verbose > 1:
                feed_dict = {
                    self.x_input: x_data[:10, :].toarray(),
                    self.y_input: y_data[:10],
                    self.keep_prob: 1 # Change dropout rate here, e.g. 0.5
                }
                predictions = self.sess.run(self.y_output, feed_dict=feed_dict)
                labels = y_train[:10]
                for i in range(len(predictions)):
                    print(
                        "%i prediction: %1.5f target: %1.5f difference: %1.5f"
                        % (i,
                           predictions[i, 0],
                           labels[i, 0],
                           abs(predictions[i, 0]-labels[i, 0]))
                    )

            if abs(meanloss-lastloss) < 0.000001:
                break
            lastloss = meanloss
        return lastloss

    def predict(self, x_data):
        "x_data must be a numpy array or a CSR matrix"
        return self.sess.run(self.y_output,
                             feed_dict={self.x_input: np.float32(x_data.toarray()),
                                        self.keep_prob:1})

    def test(self, x_test, y_test):
        "x_test and y_test must be a numpy array or a CSR matrix"
        batch_size = 128
        x_data = x_test#.tocsr()
        batches = range(0, x_data.shape[0], batch_size)
        alllosses = []
        lastbatch = 0
        for (batch_i, batch) in enumerate(batches):
            batch = batches[batch_i]
            if batch == 0:
                continue
            feed_dict = {self.x_input: x_data[lastbatch:batch, :].toarray(),
                         self.y_input: y_test[lastbatch:batch],
                         self.keep_prob: 1}
            alllosses.append(self.sess.run(self.loss, feed_dict))
            lastbatch = batch
        return np.mean(alllosses)

class SimNNR:
    """NN regressor with sentence comparison"""
    def __init__(self, layer1=50, batch_size=128, comparison=compare.SimMul()):
        self.layer1 = layer1
        self.batch_size = batch_size
        self.comparison = comparison
        self.name = "NNR-%s-relu(%i)" % (self.comparison.name, self.layer1)

    def __build_graph__(self, dimensions, learningrate=0.001):
        graph = tf.Graph()
        with graph.as_default():

            self.x_input = tf.placeholder(tf.float32,
                                          shape=(None, dimensions))
            self.q_input = tf.placeholder(tf.float32,
                                          shape=(None, dimensions))
            self.y_input = tf.placeholder(tf.float32, shape=(None, 1))
            self.keep_prob = tf.placeholder(tf.float32)

            sim = self.comparison.compare(self.x_input, self.q_input, dimensions)
            sim_layer = tf.concat(1, [self.x_input, sim])

            w_relu = tf.Variable(tf.truncated_normal([dimensions + self.comparison.size,
                                                      self.layer1],
                                                      stddev=0.1))
            b_relu = tf.Variable(tf.constant(0.1, shape=[self.layer1]))
            relu_layer = tf.nn.relu(tf.matmul(sim_layer, w_relu)+b_relu)
            relu_drop = tf.nn.dropout(relu_layer, self.keep_prob)

            # Output layer
            # w_matrix3 = tf.Variable(tf.random_uniform([layer2,1], -1.0, 1.0))
            w_out = tf.Variable(tf.truncated_normal([self.layer1, 1],
                                                    stddev=0.1))
            b_out = tf.Variable(tf.constant(0.1, shape=[1]))
            self.y_output = tf.matmul(relu_drop, w_out)+b_out


            self.loss = tf.reduce_mean(tf.square(self.y_output - self.y_input))
    #        optimizer = tf.train.GradientDescentOptimizer(learningrate)
            optimizer = tf.train.AdamOptimizer(learningrate)
            self.train = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        return graph
        
    def restore(self, savepath, dimensions):
        graph = self.__build_graph__(dimensions)
        self.sess = tf.Session(graph=graph)
        self.saver.restore(self.sess, savepath)
        print("Model restored from file: %s" % savepath)

    def fit(self, x_train, q_train, y_train, verbose=1, learningrate=0.001, nb_epoch=20,
                  dropoutrate=0.5, savepath=None):
        "x_train, q_train and y_train must be a numpy array or a CSR matrix"
        
        assert x_train.shape == q_train.shape

        if verbose > 0:
            print("%s Learning rate: %f" % (self.name, learningrate))

        graph = self.__build_graph__(x_train.shape[1], learningrate)

        self.sess = tf.Session(graph=graph)
        self.sess.run(self.init)

        lastloss = 0
        random.seed(1234)
        indices = list(range(x_train.shape[0]))
        for step in range(0, nb_epoch):
            random.shuffle(indices) # shuffle the data in each epoch
            lastbatch = 0
            batches = list(range(0, x_train.shape[0], self.batch_size))
            if batches[-1] != x_train.shape[0]:
                # Add last batch
                batches.append(x_train.shape[0])
            if verbose > 0:
                print("Epoch %i/%i" % (step, nb_epoch))
            allloss = []
            for (batch_i, batch) in enumerate(batches):
                #if batch_i > 100:
                #    break
                #batch = batches[batch_i]
                if batch == 0:
                    continue
                if isinstance(x_train, np.ndarray):
                    feed_dict = {
                        self.x_input: x_train[indices[lastbatch:batch], :],
                        self.q_input: q_train[indices[lastbatch:batch], :],
                        self.y_input: y_train[indices[lastbatch:batch]],
                        self.keep_prob: dropoutrate
                    }
                else:
                    feed_dict = {
                        self.x_input: x_train[indices[lastbatch:batch], :].todense(),
                        self.q_input: q_train[indices[lastbatch:batch], :].todense(),
                        self.y_input: y_train[indices[lastbatch:batch]],
                        self.keep_prob: dropoutrate
                    }
                    
                self.sess.run(self.train, feed_dict=feed_dict)
                thisloss = self.sess.run(self.loss,
                                         feed_dict=feed_dict)
                allloss.append(thisloss)
                if verbose > 1:
                    centile = batch_i*100/len(batches)
                    status_bar = "["+"="*int(centile/10)+" "*(10-int(centile/10))+"]"
                    print("%s %i%%  Batch Loss %f Mean Loss %f\r"
                          % (status_bar, centile, thisloss, np.mean(allloss)),
                          end='')
                    sys.stdout.flush()
                #time.sleep(0.5)
                lastbatch = batch
            if isinstance(x_train, np.ndarray):
                feed_dict = {
                    self.x_input: x_train[indices[batches[-2]:batches[-1]], :],
                    self.q_input: q_train[indices[batches[-2]:batches[-1]], :],
                    self.y_input: y_train[indices[batches[-2]:batches[-1]]],
                    self.keep_prob: 1 # Change dropout rate here, e.g. 0.5
                }
            else:
                feed_dict = {
                    self.x_input: x_train[indices[batches[-2]:batches[-1]], :].todense(),
                    self.q_input: q_train[indices[batches[-2]:batches[-1]], :].todense(),
                    self.y_input: y_train[indices[batches[-2]:batches[-1]]],
                    self.keep_prob: 1 # Change dropout rate here, e.g. 0.5
                }
                
            currentloss = self.sess.run(self.loss, feed_dict=feed_dict)
            meanloss = np.mean(allloss)
            if verbose > 0:
                status_bar = "["+"="*10+"]"
                print("%s 100%% Batch Loss %f Mean Loss %f" % (status_bar,
                                                               currentloss,
                                                               meanloss))
#            if verbose > 2:
#                feed_dict = {
#                    self.x_input: x_data[:10, :],
#                    self.y_input: y_train[:10],
#                    self.keep_prob: 1  # Change dropout rate here, e.g. 0.5
#                }
#                predictions = self.sess.run(self.y_output, feed_dict=feed_dict)
#                labels = y_train[:10]
#                for i in range(10):
#                    print(
#                        "%i prediction: %1.5f target: %1.5f difference: %1.5f"
#                        % (i,
#                           predictions[i, 0],
#                           labels[i, 0],
#                           abs(predictions[i, 0]-labels[i, 0]))
#                    )

#            if abs(meanloss-lastloss) < 0.000001:
#                break
            lastloss = meanloss
            
        if savepath:
            save_path = self.saver.save(self.sess, savepath)
            print("Model saved in file: %s" % save_path)
            
        return lastloss
    
    def predict(self, x_data, q_data):
        "x_data and q_data must be a numpy array or a CSR matrix"
        
        batches = list(range(0, x_data.shape[0], self.batch_size))
        if batches[-1] != x_data.shape[0]:
            batches.append(x_data.shape[0])
        all_y = []
        lastbatch = 0
        for (batch_i, batch) in enumerate(batches):
            batch = batches[batch_i]
            if batch == 0:
                continue
            if isinstance(x_data, np.ndarray):
                feed_dict = {self.x_input: x_data[lastbatch:batch, :],
                             self.q_input: q_data[lastbatch:batch, :],
                             self.keep_prob: 1}
            else:
                feed_dict = {self.x_input: x_data[lastbatch:batch, :].todense(),
                             self.q_input: q_data[lastbatch:batch, :].todense(),
                             self.keep_prob: 1}
            all_y.append(self.sess.run(self.y_output, feed_dict))
            lastbatch = batch
        return np.vstack(all_y)

#        return self.sess.run(
#            self.y_output,
#            feed_dict={self.x_input: np.float32(x_data),
#                       self.q_input: np.float32(q_data),
#                       self.keep_prob:1}
#        )

    def test(self, x_test, q_test, y_test):
        "x_test, q_test and y_test must be a numpy array or a CSR matrix"
            
        x_data = x_test#.tocsr()
        q_data = q_test
        batches = list(range(0, x_data.shape[0], self.batch_size))
        if batches[-1] != x_data.shape[0]:
            batches.append(x_data.shape[0])
        alllosses = []
        lastbatch = 0
        for (batch_i, batch) in enumerate(batches):
            batch = batches[batch_i]
            if batch == 0:
                continue
            if isinstance(x_data, np.ndarray):
                feed_dict = {self.x_input: x_data[lastbatch:batch, :],
                             self.q_input: q_data[lastbatch:batch, :],
                             self.y_input: y_test[lastbatch:batch],
                             self.keep_prob: 1}
            else:
                feed_dict = {self.x_input: x_data[lastbatch:batch, :].todense(),
                             self.q_input: q_data[lastbatch:batch, :].todense(),
                             self.y_input: y_test[lastbatch:batch],
                             self.keep_prob: 1}
                
            alllosses.append(self.sess.run(self.loss, feed_dict))
            lastbatch = batch
        return np.mean(alllosses)

def main():
    """Main function"""
    from sklearn import svm
    from sklearn import linear_model
    import scipy.sparse
    from matplotlib import pyplot as plt
    random.seed(1234)
    x_indices = list(range(-2560, 2560, 2))
    random.shuffle(x_indices)
    train_x = [[float(x)/500] for x in x_indices]
    print(train_x[:10])
    train_y = [np.exp(x[0]) for x in train_x]
    max_y = np.max(train_y)
    train_y = [[x/max_y+np.random.randn()*0.1] for x in train_y]
    print(train_y[:10])
    test_x = [[float(x)/50] for x in range(-255, 257, 2)]
    test_y = [[np.exp(x[0])/max_y+np.random.randn()*0.1] for x in test_x]

    print("Linear Regression")
    sk_linear_regression = linear_model.LinearRegression()
    sk_linear_regression.fit(train_x, train_y)
    predicted_y = sk_linear_regression.predict(test_x)
    plt.subplot(2, 3, 1)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, predicted_y)
    plt.title("SK Linear")

    print("Support Vector Regression")
    sk_svr = svm.SVR(kernel='rbf', C=100)
    sk_svr.fit(train_x, [y[0] for y in train_y])
    predicted_y = sk_svr.predict(test_x)
    plt.subplot(2, 3, 2)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, predicted_y)
    plt.title("SK SVR")

    nn_linear_regression = LinearRegression()
    print(nn_linear_regression.name)
    nn_linear_regression.fit(scipy.sparse.csr_matrix(train_x),
                             np.array(train_y),
                             verbose=1,
                             learningrate=0.01)
    nn_linear_regression.test(scipy.sparse.csr_matrix(test_x), test_y)
    predicted_y = nn_linear_regression.predict(scipy.sparse.csr_matrix(test_x))
    plt.subplot(2, 3, 4)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, predicted_y)
    plt.title(nn_linear_regression.name)

    nn_regression = SingleNNR(layer1=5)
    print(nn_regression.name)
    nn_regression.fit(scipy.sparse.csr_matrix(train_x),
                      np.array(train_y),
                      verbose=1,
                      learningrate=0.01)
    nn_regression.test(scipy.sparse.csr_matrix(test_x), test_y)
    predicted_y = nn_regression.predict(scipy.sparse.csr_matrix(test_x))
    plt.subplot(2, 3, 3)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, predicted_y)
    plt.title(nn_regression.name)

    nn_single_regression = SingleNNR(layer1=50)
    print(nn_single_regression.name)
    nn_single_regression.fit(scipy.sparse.csr_matrix(train_x),
                             np.array(train_y),
                             verbose=1,
                             learningrate=0.01)
    nn_single_regression.test(scipy.sparse.csr_matrix(test_x),
                              test_y)
    predicted_y = nn_single_regression.predict(scipy.sparse.csr_matrix(test_x))
    plt.subplot(2, 3, 5)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, predicted_y)
    plt.title(nn_single_regression.name)

#    dnnr = DoubleNNR(layer1=5, layer2=5)
    nn_single_regression = SingleNNR(layer1=500)
    nn_single_regression.fit(scipy.sparse.csr_matrix(train_x),
                             np.array(train_y),
                             verbose=1,
                             learningrate=0.01)
    nn_single_regression.test(scipy.sparse.csr_matrix(test_x),
                              test_y)
    predicted_y = nn_single_regression.predict(scipy.sparse.csr_matrix(test_x))
    plt.subplot(2, 3, 6)
    plt.scatter(test_x, test_y)
    plt.scatter(test_x, predicted_y)
    plt.title(nn_single_regression.name)

    plt.show()

    sys.exit()

    import pickle
    with open('../trainfeatures1.pickle') as f:
        (trainfeatures, trainscores, _tfidf) = pickle.load(f)
    with open('../testfeatures1.pickle') as f:
        (testfeatures, testscores, _tfidf) = pickle.load(f)
    nn_linear_linear_regression = LinearRegression()
#    trainloss = nn_linear_regression.fit(trainfeatures.tocsr()[:1000],trainscores[:1000])
    trainloss = nn_linear_regression.fit(trainfeatures.tocsr(), trainscores)
    testloss = nn_linear_regression.test(testfeatures.tocsr(), testscores)
    print("MMR train: %1.5f test: %1.5f" % (trainloss, testloss))

if __name__ == "__main__":
    import sys
    import random
    import doctest
    doctest.testmod()
    main()
