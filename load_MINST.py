import _pickle as cPickle
import gzip
import numpy as np
#..


def reformat(labels):
    num_labels = 10
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels


f = gzip.open('mnist.pkl.gz', 'rb')

save = cPickle.load(f,encoding='latin1')
train_dataset = save[0][0]
train_labels = reformat(save[0][1])
raw_train_labels = save[0][1]

valid_dataset = save[1][0]
valid_labels = reformat(save[1][1])
raw_valid_labels = save[1][1]

test_dataset = save[2][0]
test_labels = reformat(save[2][1])
raw_test_labels = save[2][1]

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


f.close()