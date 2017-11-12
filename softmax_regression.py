from utilities import*

def add_ones(X):
    return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))

def train_log_regression(X, y, validation_data_input, valid_labels, raw_train_labels, raw_valid_labels):
    X = add_ones(X) # Bias term
    maxiter = 50
    batch_size = 20
    # of inputs
    n = X.shape[1]
    # of data
    m = X.shape[0]
    curr_weight = np.random.rand(n, len(y[0])) # Initialise random weights
    lmbda = 0.01
    alpha = 0.01
    max_error = 0.1
    loss = 10
    err_iteration = []
    train_accuracy = []
    validation_accuracy = []
    for iteration in range(maxiter):
        start = 0
        for i in range(int(m/batch_size)):

            # 10 x 784 probability via softmax
            out_probs = get_probabilities(X[start:start+batch_size], curr_weight)

            grad = (1.0/batch_size) * np.dot(X[start:start+batch_size].T, (out_probs - y[start:start+batch_size]))
            # g0 = grad[0]
            grad += ((lmbda * curr_weight) / batch_size)
            # grad[0] = g0
            curr_weight -= alpha * grad

            # calculate the magnitude of the gradient and check for convergence
            loss = calculate_entropy_loss(X[start:start+batch_size], curr_weight, y)
            start += batch_size
        err_iteration.append(loss)

        pred_output_train = np.dot(X, curr_weight)
        single_train_accu = accuracy(raw_train_labels, one_hot_encoding(pred_output_train))
        train_accuracy.append(single_train_accu)

        pred_output_valid = np.dot(add_ones(validation_data_input), curr_weight)
        single_valid_accu = accuracy(raw_valid_labels, one_hot_encoding(pred_output_valid))
        validation_accuracy.append(single_valid_accu)

        if np.abs(loss)< max_error or math.isnan(loss):
            break
    return curr_weight, err_iteration, train_accuracy, validation_accuracy

