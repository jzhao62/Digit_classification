# import numpy as np
# import _pickle as cPickle
# import gzip

#import matplotlib.pyplot as plt
#import tensorflow as tf
from load_data import*
from softmax_regression import*
from SNN import*
from utilities import*
from CNN import*



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
        print ("USPS Accuracy - Single Layer NN: ", usps_accuracy_nn)
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
