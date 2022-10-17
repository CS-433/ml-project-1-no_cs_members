# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


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
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************

    new_x = np.reshape(x,(x.shape[0],1))
    #new_x = x
    poly = np.ones(np.shape(new_x))
    
    for j in range(1,degree+1):
        # poly = np.append(poly,x**j,axis=1)
        poly = np.concatenate((poly,new_x**j),axis=1)
    
    return poly