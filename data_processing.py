import numpy as np


def standardize_data_999removed(y,tx):
    """remove all points (rows) with missing data
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) N is the number of samples, D is the number of features
    
    Returns:
        y_new: reduced y
        tx_new: reduced x
        means: array of the mean of each feature 
        std_dev: array of the standard deviation of each feature
    """
    idx_incomplete_points = np.nonzero(tx[:,4]==-999)
    tx_new = np.delete(tx,idx_incomplete_points,0)
    y_new = np.delete(y,idx_incomplete_points)
    means = np.mean(tx_new, axis=0)
    means = means * np.ones(np.shape(tx_new))
    tx_new_std = tx_new - means
    std_dev = np.std(tx_new, axis=0)
    std_dev = std_dev * np.ones(np.shape(tx_new))
    tx_new_std = tx_new_std / std_dev      
    y_new = np.reshape(y_new,[len(y_new),1])
    return y_new, tx_new, means, std_dev


def standardize_data_999mean(tx):
    """replace missing data in tx in a feature (column) by the mean of the 
        feature across dataset
    
    Args:
        tx: shape=(N,P) N is the number of samples, D is the number of features
    
    Returns:
        tx: modified tx with missing data replaced by means
        means: array of the mean of each feature 
        std_dev: array of the standard deviation of each feature
    """

    idx_incomplete_points = np.nonzero(tx[:,4]==-999)
    tx_rem = np.delete(tx,idx_incomplete_points,0)
    
    means = np.mean(tx_rem, axis=0)
    std_dev = np.std(tx_rem - means, axis=0)

    for i in range(tx.shape[1]):
        feature = tx[:,i]
        feature[feature==-999] = means[i]
        tx[:,i] = feature
    
    tx = (tx - means) / std_dev
    return tx, means, std_dev


def split_data(y,tx,ratio):
    """split a data set in a training part and a test part
    with a given ratio
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        ratio: scalar, indicates amount of training data
    
    Returns:
        y_tr, x_tr: training data
        y_te, x_te: test data
    """
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    x_tr = tx[index_tr]
    x_te = tx[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return y_tr, x_tr, y_te, x_te

def add_w0(tx,N):
    tx = np.concatenate((tx,np.ones([N,1])),axis=1)  
    return tx
