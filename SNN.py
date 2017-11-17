from utilities import*

def train_single_layer_nn(train_dataset, train_labels, validation_data_input, raw_valid_labels):
    print('successfully called training method in SNN');
    #  TODO: change this to smaller scale if you want see faster result (but will be more erroronous)
    train_size= 5000
    test_size=10000
    input_size=784
    output_size=10
    hidden_layer_size = 500
    neural_neta = 0.5

    # batch size for GD
    mini_batch_size = 1000
    mini_b_start = 1

    # weights 1 input(786 Units) -> HL (500 Units)
    hidden_wts = np.random.normal(0, 0.2, (hidden_layer_size, input_size))

    # weights 2 HL (500 Units) -> output (10 Units)
    neural_out_wts = np.random.normal(0, 0.2,(output_size, hidden_layer_size))

    #     TODO: wtf is this
    hidden_wts = hidden_wts/len(hidden_wts[1])
    neural_out_wts = neural_out_wts/len(neural_out_wts[1])


    n = np.zeros(mini_batch_size)
    targetValues= train_labels.T

    validation_accuracy = []
    train_losses = []
    num_iter=0
    while(num_iter < 5):

        mini_b_stop = min(train_size,mini_b_start+mini_batch_size-1);
        curr_train_design_mat = train_dataset[mini_b_start:mini_b_stop]
        curr_train_output_k_format = train_labels[mini_b_start:mini_b_stop]
        curr_train_size = len(curr_train_design_mat)

        # miniBatch GD, until model is trained
        for j in range(0,mini_batch_size-1):
            train_line = curr_train_design_mat[j]
            forward_prop_one = np.dot(hidden_wts,train_line)
            hidden_out = sigmoid(forward_prop_one);
            forward_prop_two = np.dot(neural_out_wts,hidden_out)
            output = sigmoid(forward_prop_two)
            train_out_line = curr_train_output_k_format[j]

            #  Error propagation 2 steps.
            second_err = np.multiply(np.multiply(output,(1 - output)),(output - train_out_line))
            hidden_err = np.multiply(np.multiply(hidden_out,(1 - hidden_out)),np.dot(neural_out_wts.T,second_err))

            neural_out_wts = neural_out_wts - 0.5 * np.dot(np.vstack(second_err),np.vstack(hidden_out).T)
            hidden_wts = hidden_wts - 0.5 * np.dot(np.vstack(hidden_err),np.vstack(train_line).T)
        train_losses.append(np.sum(output - train_out_line))
        mini_b_start = mini_b_start+mini_batch_size

        print('model training finished, ending at: #',mini_b_start);

        # generate model accuracy based on validation data
        if(mini_b_start > train_size):
            print(num_iter)
            trained_val = np.zeros(len(validation_data_input))
            test_input = validation_data_input
            test_pred = []
            for i in range(len(test_input)):
                test_inp_line = test_input[i]
                hidden_inp = np.dot(hidden_wts, test_inp_line)
                hidden_out = 1 / (1 + np.exp(-1 * hidden_inp))
                second_inp = np.dot(neural_out_wts, hidden_out)
                neural_out_line = 1 / (1 + np.exp(-1 * second_inp))
                test_pred.append(neural_out_line)
            print("Validation accuracy: ", accuracy( raw_valid_labels, one_hot_encoding(np.array(test_pred))))
            validation_accuracy.append(accuracy(raw_valid_labels, one_hot_encoding(np.array(test_pred))))
            mini_b_start = 1
            num_iter = num_iter + 1
    return hidden_wts, neural_out_wts, validation_accuracy, train_losses
def evaluate_nn(test_dataset, raw_test_labels, hidden_wts, neural_out_wts):
    test_pred = []
    for i in range(len(test_dataset)):
        test_inp_line = test_dataset[i]
        hidden_inp = np.dot(hidden_wts, test_inp_line)
        hidden_out = 1/(1 + np.exp(-1 * hidden_inp))
        second_inp = np.dot(neural_out_wts,hidden_out)
        neural_out_line = 1/(1 + np.exp(-1 * second_inp))
        test_pred.append(neural_out_line)
    test_accuracy = accuracy(raw_test_labels,one_hot_encoding(np.array(test_pred)))
    return test_accuracy
