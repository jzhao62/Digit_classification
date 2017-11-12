
import numpy as np
import math


def get_probabilities(X, curr_weight):
    t = np.dot(X, curr_weight)
    prob_vector = softmax(t)
    return prob_vector


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
    # computed probability
    O = np.dot(X, w)
    for i in range(len(O)):
        loss += -1 * np.dot(y[i], np.log(O[i].T))
    return float(loss)/len(O)

def accuracy(y, result):
    count = 0
    for i in range(len(y)):
        if y[i] == result[i]:
            count += 1
    return float(count)/len(y)

def one_hot_encoding(t):
    return np.argmax(t, axis=1)
