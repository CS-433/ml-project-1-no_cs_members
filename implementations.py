
from helpers_stochastic_gradient import*
from helpers import*
from costs import*

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
