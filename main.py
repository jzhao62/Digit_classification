# import numpy as np
# import _pickle as cPickle
# import gzip

import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import*
from softmax_regression import*
from SNN import*
from utilities import*





def reformat_tf(dataset, labels):
    image_size = 28
    num_labels = 10
    num_channels = 1
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy_tf(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def plot_data(y_values1, label, axis_dim, xlabel, ylabel, title):
    plt.plot(y_values1, 'b-',label=label)
    plt.axis(axis_dim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    l = plt.legend()
    plt.show()


def preprocess_data_cnn_tf(train_dataset, validation_data_input, test_dataset,
                           raw_train_labels, raw_valid_labels, raw_test_labels,
                           validation_usps, validation_usps_label):
    image_size = 28
    num_labels = 10
    num_channels = 1 # grayscale
    training_inp = np.reshape(train_dataset, (50000, 28, 28))
    valid_inp = np.reshape(validation_data_input, (10000, 28, 28))
    test_inp = np.reshape(test_dataset, (10000, 28, 28))
    usps_data = np.reshape(validation_usps, (19999, 28, 28))

    train_dataset, train_labels = reformat_tf(training_inp, raw_train_labels)
    validation_data_input, valid_labels = reformat_tf(valid_inp, raw_valid_labels)
    test_dataset, test_labels = reformat_tf(test_inp, raw_test_labels)
    usps_dataset, usps_labels = reformat_tf(usps_data, validation_usps_label)

    return train_dataset, train_labels, validation_data_input, valid_labels, test_dataset, test_labels, usps_dataset, usps_labels


def train_cnn_model(train_dataset, train_labels,
                    validation_data_input, valid_labels,
                    test_dataset, test_labels,
                    usps_dataset, usps_labels):
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64
    beta_regul = 1e-3
    drop_out = 0.5
    image_size = 28
    num_labels = 10
    num_channels = 1
    graph = tf.Graph()

    with graph.as_default():
          # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(validation_data_input)
        tf_test_dataset = tf.constant(test_dataset)
        tf_usps_dataset = tf.constant(usps_dataset)
        beta_regul = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2
        layer3_weights = tf.Variable(tf.truncated_normal([size3 * size3 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
        layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        def model(data, keep_prob):
            # C1 input 28 x 28
            conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
            bias1 = tf.nn.relu(conv1 + layer1_biases)
            # S2 input 24 x 24
            pool2 = tf.nn.max_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
            # C3 input 12 x 12
            conv3 = tf.nn.conv2d(pool2, layer2_weights, [1, 1, 1, 1], padding='VALID')
            bias3 = tf.nn.relu(conv3 + layer2_biases)
            # S4 input 8 x 8
            pool4 = tf.nn.max_pool(bias3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
            # F5 input 4 x 4
            shape = pool4.get_shape().as_list()
            reshape = tf.reshape(pool4, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            # F6
            drop5 = tf.nn.dropout(hidden5, keep_prob)
            hidden6 = tf.nn.relu(tf.matmul(hidden5, layer4_weights) + layer4_biases)
            drop6 = tf.nn.dropout(hidden6, keep_prob)
            return tf.matmul(drop6, layer5_weights) + layer5_biases

        logits = model(tf_train_dataset, drop_out)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta_regul * (tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer4_weights)+tf.nn.l2_loss(layer5_weights))
        
        # Optimizer.
        learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.85, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))
        usps_prediction = tf.nn.softmax(model(tf_usps_dataset, 1.0))

    num_steps = 20001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
              print('Minibatch loss at step %d: %f' % (step, l))
              print('Minibatch accuracy: %.1f%%' % accuracy_tf(predictions, batch_labels))
              print('Validation accuracy: %.1f%%' % accuracy_tf(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy_tf(test_prediction.eval(), test_labels))
        print('USPS data accuracy: %.1f%%' % accuracy_tf(usps_prediction.eval(), usps_labels))

def main():
    ''' Fetch MNIST Data and USPS Data'''
    train_dataset, train_labels, raw_train_labels, \
    validation_data_input, valid_labels, raw_valid_labels, \
    test_dataset, test_labels, raw_test_labels = import_MNIST()
    print("MNIST Data fetched, done")
    print('------------------------------')
    validation_usps, validation_usps_label = import_USPS()

    print('usps validation_image: ', validation_usps.shape);
    print('usps validation label: ', validation_usps_label.shape)
    print ("USPS Data fetched, done")
    print('------------------------------')

    # TODO Generate result for Linear Regression model
    if(True):
        ''' Train Logistic Regression Classifier'''
        weights_lr, err_iteration_lr, train_accuracy_lr, validation_accuracy_lr = train_log_regression(train_dataset, train_labels,
                                                                                                       validation_data_input, valid_labels,
                                                                                                       raw_train_labels, raw_valid_labels)
        print ("------------LR model trained------------")
        plot_data(err_iteration_lr, 'loss', [0, 100, -2, -4], 'Iterations', 'Cross entropy loss', 'Loss change over iterations')
        plot_data(train_accuracy_lr, 'loss', [0, 100, 0.8, 1], 'Iterations', 'Training Accuracy', 'Training accuracy change over iterations')
        plot_data(validation_accuracy_lr, 'loss', [0, 100, 0.8, 1], 'Iterations', 'Validation Accuracy', 'Validation accuracy change over iterations')
        pred_output_train_lr = np.dot(add_ones(train_dataset), weights_lr)
        print ("Training Set Accuracy - Logistic Regression: ", accuracy(raw_train_labels, one_hot_encoding(pred_output_train_lr)))
        pred_output_valid_lr = np.dot(add_ones(validation_data_input), weights_lr)
        print ("Validation Set Accuracy - Logistic Regression: ", accuracy(raw_valid_labels, one_hot_encoding(pred_output_valid_lr)))
        pred_output_test_lr = np.dot(add_ones(test_dataset), weights_lr)
        print ("Test Set Accuracy - Logistic Regression: ", accuracy(raw_test_labels, one_hot_encoding(pred_output_test_lr)))

        ''' Performance of LR on USPS Data'''
        pred_output_usps_lr = np.dot(add_ones(validation_usps), weights_lr)
        print ("USPS Accuracy - Logistic Regression: ", accuracy(validation_usps_label, one_hot_encoding(pred_output_usps_lr)))
    # TODO Generate result for SNN
    if(False):
        hidden_wts_nn, out_weights_nn, validation_accuracy_nn, train_losses_nn = train_single_layer_nn(train_dataset, train_labels, validation_data_input, raw_valid_labels)
        print ("SNN model trained")
        ''' Performance of Single Layer NN on MNIST Data '''
        plot_data(train_losses_nn, 'loss', [0, 100, -5, 5], 'Iterations', 'Training Loss', 'Training Losses over epochs: Single layer NN')
        plot_data(validation_accuracy_nn, 'accuracy', [0, 4, 0.9, 1], 'Iterations', 'Validation Accuracy', 'Validation accuracy over iterations Single layer NN')
        test_accuracy_nn = evaluate_nn(test_dataset, raw_test_labels, hidden_wts_nn, out_weights_nn)
        print ("Test set Accuracy - Single Layer NN: ", test_accuracy_nn)

        ''' Performance of Single Layer NN on USPS Data '''
        usps_accuracy_nn = evaluate_nn(validation_usps, validation_usps_label, hidden_wts_nn, out_weights_nn)
    # TODO Generate result for CNN
    if(False):
        ''' Train a Convolutional Neural Network'''
        print ("Training a CNN")
        train_dataset_cnn, train_labels_cnn, \
        valid_dataset_cnn, valid_labels_cnn, \
        test_dataset_cnn, test_labels_cnn, \
        usps_dataset_cnn, usps_labels_cnn = preprocess_data_cnn_tf(train_dataset, validation_data_input, test_dataset,
                                                                   raw_train_labels, raw_valid_labels, raw_test_labels,
                                                                   validation_usps, validation_usps_label)


        train_cnn_model(train_dataset_cnn, train_labels_cnn,
                        valid_dataset_cnn, valid_labels_cnn,
                        test_dataset_cnn, test_labels_cnn,
                        usps_dataset_cnn, usps_labels_cnn)

    print("------------Done--------------")

main()