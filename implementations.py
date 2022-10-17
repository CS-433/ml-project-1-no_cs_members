
from helpers_stochastic_gradient import*
from helpers_gradient import*
from helpers import*
from costs import*
import numpy as np

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    N = y.shape[0]
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y - w.dot(tx.T)                    #mse = (1/(2*N)) * np.sum((y-w[0]-w[1]*tx[:,1])**2)
    mse = (1/(2*N))*e.T.dot(e)
    # ***************************************************
    return w, mse

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma) : 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute the gradient 
        grad = compute_gradient(y, tx, w)
       
        # update w by gradient descent
        w = w - gamma * grad
        # compute loss for this w
        loss = compute_loss(y,tx,w)
      

    return w, loss
    
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

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    
    n = tx.shape[0]
    d = tx.shape[1]
    w = np.linalg.solve(np.dot(tx.T,tx) + lambda_*2*n * np.identity(d), np.dot(tx.T,y))
    loss = compute_loss(y,tx,w)
    
    return w, loss
