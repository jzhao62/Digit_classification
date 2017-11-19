
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics

def get_probabilities(X, curr_weight):
    t = np.dot(X, curr_weight)
    return softmax(t)


def softmax(t):
    prob_matrix = []
    for i in range(len(t)):
        prob_vector = []
        sum_exp = 0
        for j in t[i]:
            sum_exp += np.exp(j)
        for k in t[i]:
            prob_vector.append(float(np.exp(k))/sum_exp)
        prob_matrix.append(prob_vector)
    return np.array(prob_matrix)

def sigmoid(raw):
    ans = 1/(1 + np.exp(-1 * raw));
    return ans;


def calculate_entropy_loss(X, w, y):
    loss = 0
    temp = np.dot(X, w)
    for i in range(len(temp)):
        loss += -1 * np.dot(y[i], np.log(temp[i].T))
    return float(loss)/len(temp)

def accuracy(y, result):
    count = 0
    for i in range(len(y)):
        if y[i] == result[i]:
            count += 1
    return float(count)/len(y)

def one_hot_encoding(t):
    return np.argmax(t, axis=1)

def plot_data(y_values1, graph_label, axis_dim, xlabel_name, ylabel_name, title):
    plt.plot(y_values1, 'g', label=graph_label)
    plt.axis(axis_dim)
    plt.ylabel(ylabel_name)
    plt.xlabel(xlabel_name)
    plt.title(title)
    l = plt.legend()
    plt.show()

def plot_confusion_matrix(cm,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    classes = [0,1,2,3,4,5,6,7,8,9]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_train_validation_accuracy(train_accu, valid_accu, range, label1, label2,  title):
    y1 = train_accu
    y2 = valid_accu
    xrange = range
    plt.plot(y1, 'g', label=label1)
    plt.plot(y2, 'r', label=label2)
    # plt.axis(xrange)
    plt.ylabel("accu")
    plt.xlabel("Epochs")
    plt.title(title)
    l = plt.legend()
    plt.show()
