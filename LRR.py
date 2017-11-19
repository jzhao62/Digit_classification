from utilities import*

def add_bias(X):
    return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))

def train_log_regression(X, Y, validation_data_input, valid_labels, raw_train_labels, raw_valid_labels):
    X = add_bias(X) # Bias term
    V = add_bias(validation_data_input)
    # of inputs
    # of data
    n = X.shape[1]
    m = X.shape[0]
    curr_weight = np.random.rand(n, len(Y[0])) # Initialise random weights
    loss_record = []
    train_accuracy = []
    validation_accuracy = []

    batch_size = 10
    lmbda = 0.01
    alpha = 0.01
    loss = 0

    for itr in range(30):
        s = 0
        for i in range(int(m/batch_size)):

            # 10 x 784 probability via softmax
            out_probs = get_probabilities(X[s:s+batch_size], curr_weight)

            curr_x= X[s:s+batch_size];
            curr_y = Y[s:s+batch_size]

            grad = (1.0/batch_size) * np.dot(curr_x.T, (out_probs - curr_y))
            grad += ((lmbda * curr_weight) / batch_size)
            curr_weight -= alpha * grad
            loss = calculate_entropy_loss(curr_x, curr_weight, Y)
            s += batch_size

        loss_record.append(abs(loss))
        print(itr, " ", abs(loss))
        pred_output_train = np.dot(X, curr_weight)
        single_train_accu = accuracy(raw_train_labels, one_hot_encoding(pred_output_train))
        train_accuracy.append(single_train_accu)

        pred_output_valid = np.dot(V, curr_weight)
        single_valid_accu = accuracy(raw_valid_labels, one_hot_encoding(pred_output_valid))
        validation_accuracy.append(single_valid_accu)

    return curr_weight, loss_record, train_accuracy, validation_accuracy



def evaluate_LR(weights_lR, MNIST_test_image, MNIST_test_label, USPS_test_image, USPS_test_label):

    pred_output_test_lr = np.dot(add_bias(MNIST_test_image), weights_lR)
    print("Test Set Accuracy(MNIST) - LR: ", accuracy(MNIST_test_label, one_hot_encoding(pred_output_test_lr)))

    pred_output_usps_lr = np.dot(add_bias(USPS_test_image), weights_lR)
    usps_accu = accuracy(USPS_test_label, one_hot_encoding(pred_output_usps_lr))
    print("USPS Accuracy - LR: ", usps_accu)

    cnf = metrics.confusion_matrix(USPS_test_label, one_hot_encoding(pred_output_usps_lr))

    plt.figure()
    plot_confusion_matrix(cnf, title='Confusion matrix on USPS (Softmax LR), Accuracy: %.1f%%' % (usps_accu * 100))
    plt.show()

