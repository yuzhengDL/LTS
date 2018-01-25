import numpy as np


def load_dataset(file, load_train=True):

    if load_train:
        train_caps = np.load(file + '/f8k_train_rnn.npy')
	train_ims = np.load(file + '/f8k_train_ims(f30k).npy')
    else:
        train_caps = None
	train_ims = None

    #dev_caps = np.load(file + '/f8k_dev_rnn.npy')
    #dev_ims = np.load(file + '/f8k_dev_ims.npy')

    test_caps = np.load(file + '/f8k_test_rnn.npy')
    test_ims = np.load(file + '/f8k_test_ims(f30k).npy')

    return (train_caps, train_ims), (test_caps, test_ims)

# Get batch data from training set
def get_batch_data(data):
    ims = []
    pos_ls = []
    neg_ls = []
    for i in range(len(data)):
        tup = data[i]
        ims.append(tup[0])
	pos_ls.append(tup[1])
	neg_ls.append(tup[2])
    return ims, pos_ls, neg_ls
