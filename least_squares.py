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