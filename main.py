
from load_data import*
from LRR import*
from SNN import*
from CNN import*
from utilities import*

def main():
    train_dataset, train_labels, raw_train_labels, \
    validation_data_input, valid_labels, raw_valid_labels, \
    test_dataset, test_labels, raw_test_labels = import_MNIST()
    print("MNIST Data fetched, done")
    print('------------------------------')

    validation_usps, validation_usps_label = load_usps('usps_test_image.pkl','usps_test_label.pkl')
    print('usps validation_image: ', validation_usps.shape);
    print('usps validation label: ', validation_usps_label.shape)
    print ("USPS Data fetched, done")
    print('------------------------------')


    if(True):
        ''' LRR'''
        weights_lr, loss_record_lr, train_accuracy_lr, validation_accuracy_lr = train_log_regression(train_dataset, train_labels,
                                                                                                       validation_data_input, valid_labels,
                                                                                                       raw_train_labels, raw_valid_labels)
        print ("------------LR model trained------------")
        plot_data(loss_record_lr, 'loss', [0, 30, 0, 4], 'Epochs', 'abs loss', 'Loss change over Epochs')

        plot_train_validation_accuracy(train_accuracy_lr, validation_accuracy_lr, [0,150], "Training", "Validation", "Training & Validation accuracy")

        pred_output_train_lr = np.dot(add_bias(train_dataset), weights_lr)
        print ("Training Set Accuracy(MNIST) - LR: ", accuracy(raw_train_labels, one_hot_encoding(pred_output_train_lr)))

        evaluate_LR(weights_lr, train_dataset, raw_train_labels, validation_usps, validation_usps_label)

        print('------------------------------------------------------------------------------------------')

    if(False):
        hidden_wts_nn, out_weights_nn, validation_accuracy_nn, train_accuracy_nn, train_losses_nn = train_single_layer_nn(train_dataset, train_labels, raw_train_labels, validation_data_input, raw_valid_labels)
        print ("SNN model trained")
        plot_data(train_losses_nn, 'loss', [0, 11, -4, 4], 'Epochs', 'Loss', 'Training Losses over epochs: SNN')
        plot_train_validation_accuracy(train_accuracy_nn, validation_accuracy_nn, [0,11], "Training", "Validation", " Training & validation accuracy (SNN)")

        test_accuracy_nn, cnf_mnist = evaluate_nn(test_dataset, raw_test_labels, hidden_wts_nn, out_weights_nn,"MNIST_testSet")
        print ("Test set Accuracy - SNN: ", test_accuracy_nn)

        usps_accuracy_nn, cnf_usps = evaluate_nn(validation_usps, validation_usps_label, hidden_wts_nn, out_weights_nn, "USPS_testSet")
        print('Performance on USPS(SNN): ', usps_accuracy_nn)
        plt.figure()
        plot_confusion_matrix(cnf_usps, title='Confusion matrix on USPS (SNN), Accuracy: %.1f%%' % (usps_accuracy_nn * 100))
        plt.show()

    if(False):
        ''' Train a Convolutional Neural Network'''

        train_dataset_cnn, train_labels_cnn, \
        valid_dataset_cnn, valid_labels_cnn, \
        test_dataset_cnn, test_labels_cnn, \
        usps_dataset_cnn, usps_labels_cnn = preprocess_data_cnn_tf(train_dataset, validation_data_input, test_dataset,
                                                                   raw_train_labels, raw_valid_labels, raw_test_labels,
                                                                   validation_usps, validation_usps_label)


        loss_record = train_cnn_model(train_dataset_cnn, train_labels_cnn,
                        valid_dataset_cnn, valid_labels_cnn,
                        test_dataset_cnn, test_labels_cnn,
                        usps_dataset_cnn, usps_labels_cnn)

        print ("------------CNN model trained------------")
        plot_data(loss_record, 'Loss', [0, 10003, 0, 3.5], 'Epochs', 'Loss','Training Loss change over Epochs: CNN)')

    print('\n')
    print('\n')
    print("------------Finished--------------")

main()