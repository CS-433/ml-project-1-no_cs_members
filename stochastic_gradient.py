def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]





def compute_loss(y, tx, w):

    """Calculate the loss using MSE 

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    
    n = y.size
    txw = np.dot(tx,w)
    e = y - txw
    L = np.sum(np.square(e))
    L = L/n
    return L


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    
  
    nb_batch = 1
    gradient = np.zeros(nb_batch)
  
    for minibatch_y, minibatch_tx in batch_iter(y, tx, nb_batch):
        
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
      
    return gradient


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma) : 
    #Linear regression using stochastic gradient descent
    #Compute the stochastic gradient descent algorithm for one batch
    
    """Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: loss value (scalar) for the last iteration 
        w:  model parameters (numpy array of size ...)for the last iteration
    """
    
    w = initial_w
    
    
    for n_iter in range(max_iters):
        
        grad = compute_stoch_gradient(y,tx,w)
        w = w - gamma*grad
        loss = compute_loss (y,tx,w)
       
    return w, loss
    
