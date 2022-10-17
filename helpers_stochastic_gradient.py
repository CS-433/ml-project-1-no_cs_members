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



    
