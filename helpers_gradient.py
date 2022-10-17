def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    n = y.shape[0]
    e = y - w[0] - np.multiply(w[1],x[1])
    #dL = np.array([(-1/n)*np.sum(e), (-1/n)*np.sum(e*x[1])])
    dL = (-1/n) *np.dot(tx.T,e) 
    return dL
