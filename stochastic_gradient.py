from helpers import *

from compute_loss import *

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
    
