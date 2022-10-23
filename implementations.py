## ***************************************************
#  IMPLEMENTATIONS OF REQUIRED FUNCTIONS

## ***************************************************

import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    returns optimal weights, and mse.
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w -= gamma * compute_mse_gradient(y,tx,w)
    
    return w, compute_mse_loss(y,tx,w)



def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    returns optimal weights, and mse.

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w -= gamma * compute_stoch_mse_gradient(y,tx,w)

    return w, compute_mse_loss(y,tx,w)



def least_squares(y, tx):
    """Calculate the least squares solution.
    returns optimal weights, and mse.
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A,b)
    return w, compute_mse_loss(y,tx,w)



def ridge_regression(y,tx,lambda_):
    """implement ridge regression.
    returns optimal weights, and mse.
    
    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    A = tx.T@tx + 2*tx.shape[0]*lambda_*np.eye(tx.shape[1])
    b = tx.T@y
    w = np.linalg.solve(A,b)
    return w, compute_mse_loss(y,tx,w)



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y=0,1)

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w-gamma * calculate_logistic_gradient(y,tx,w)
    loss = calculate_logistic_loss(y,tx,w)
    return w, loss



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic Ridge regression using gradient descent (y=0,1)

    Args:
        y: shape=(N, ) N is the number of samples
        tx: shape=(N,P) D is the number of features
        initial_w: shape=(P, ). The vector of model parameters.
        max_iters: scalar
        gamma: scalar. Step-size in gradient-descent

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    w = initial_w
    for n_iter in range(max_iters):
        penalized_gradient = calculate_logistic_gradient(y,tx,w)+ 2 * lambda_ * w
        w -= gamma * penalized_gradient
    loss = calculate_logistic_loss(y,tx,w) 
    # convention: loss is always without the penalty term
    return w, loss


## ***************************************************
#  auxiliary functions 

## ***************************************************

def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return (1/(2*len(y))) * np.sum(np.square(y - tx@w))

def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    return - (tx.T@(y - tx@w))/len(y)

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


def compute_stoch_mse_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    batch_size=1
    for value in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): 
        grad = compute_mse_gradient(value[0],value[1],w)
    
    return grad

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1/(1+np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss
    """
    # assert y.shape[0] == tx.shape[0]
    # assert tx.shape[1] == w.shape[0]

    loss = -(1/len(y)) * ( y.T @ np.log(sigmoid(tx@w)) + (1-y).T @ np.log(1-sigmoid(tx@w)) )
    loss = np.sum(loss)  # to avoid nested 1-D arrays
    return loss

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)
    """
    return (1/len(y))*tx.T@(sigmoid(tx@w)-y)

#Hyperparameters optimization

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
    

def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k,initial_w, lambda_, degree ,gamma, max_iters):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)"""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)] #comprendre cette ligne ?
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    

    w, loss_tr = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
   
    loss_te =  calculate_logistic_loss(y_te, x_te, w)
    
    return loss_tr, loss_te



def cross_validation_demo(y, x, k_fold,k, initial_w, lambdas, degree ,gamma, max_iters):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    max_iters = max_iters
    
    for lambda_ in lambdas : 
        rmse_tr_k = []
        rmse_te_k = []

        for  k in range(k_fold): 
                
                tr,te = cross_validation(y, x, k_indices,k, initial_w,lambda_, degree ,gamma, max_iters)
                rmse_tr_k.append(tr)
                rmse_te_k.append(te)
        
        rmse_tr.append(np.mean(rmse_tr_k))
        rmse_te.append(np.mean(rmse_te_k))
     
    best_rmse = np.min(rmse_tr)
    best_ind= np.argmin(rmse_tr)
    best_lambda = lambdas[best_ind]
      

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse


#Use poly and find best degree

def best_degree_selection(y,x,degrees, k_fold, initial_w, lambdas, gamma,max_iters,seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated rmse
    best_lambdas = []
    best_rmses = []
    #vary degree
    for degree in degrees:
        # cross validation
        phi = build_poly(x,degree)
        
        rmse_te = []
        for lambda_ in lambdas:
            initial_w = ridge_regression(y,phi,lambda_)[0]
            rmse_te_tmp = []
            for k in range(k_fold):
                
                _, loss_te= cross_validation(y, phi, k_indices,k, initial_w,lambda_, degree ,gamma, max_iters)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))
        
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree]


# Voir a quoi peuvent servir les premières fonctions codées ?? initial_w ?
# faire test des différentes méthodes qu'on peut utiliser pour la log regression
# voir comment optimiser ?
# changer un peu les fonctions peut etre? qu'est ce qu'on recherche ? discuter






