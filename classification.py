# # ***************************************************
#  IMPLEMENTATIONS OF  ADDITIONNAL CLASSIFICATION FUNCTION

# # ***************************************************

import numpy as np
import matplotlib.pyplot as plt
from math import dist,ceil


def get_Kneighbors(x_tr, K, new_sample):
    """Finds K nearest neighbors using euclidian distance
    
    Args:
        x_tr: 
        K: 
        new_sample: 
        
    Returns:
        Kneighbors:
        Kindexes:    
    """
    samples = np.shape(x_tr)[0] 
    neighbors = np.zeros((samples,1))
    Kneighbors = np.zeros((K,1))
    Kindexes = np.zeros((K,1))
    
    for i in range (samples):
        neighbors[i] = dist(x_tr[i,:], new_sample)  #computes euclidean distance between two samples
           
    for j in range (K):
        Kneighbors[j] = min(neighbors)
        idx = np.argmin(neighbors)
        #remove for the next iteration the last smallest value:
        Kindexes[j] = idx
        neighbors = np.delete(neighbors, idx)
        
    return Kneighbors, Kindexes  




def get_prediction(Kindexes, K, y_tr):
    """Predicts y = 0 or 1 using the prediction of K-nearest neighbors
    
    Args:
        Kindexes: 
        K:  
        
    Returns:
        predictions:
        new_prediction:    
    """
    
    predictions = np.zeros((K,1))
    
    for m in range (K):
        predictions[m] = y_tr[np.int(Kindexes[m])]  #warning because of forced int()

    if sum(predictions) >= ceil(K/2):
        new_prediction = 1
    else:
        new_prediction = 0
        
    return predictions, new_prediction



def get_accuracy(y_predictions, y_te):
    """Checks whether prediction are accurate by compraing with y_te
    
    Args: 
        predictions:
        y_te:
    
    Returns:
        len(good_guess):
        len(bad_guess):
    """ 
    
    difference = (y_predictions-y_te)
    good_guess = difference[difference==0]
    bad_guess = difference[difference!=0]
    accuracy = len(good_guess)/(len(good_guess)+len(bad_guess))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(y_predictions.shape[0]):
        if difference[i] == 1:
            FP +=1
        if difference[i] == -1:
            FN +=1
        else :
            if y_predictions[i] == 1:
                TP +=1
            else:
                TN +=1
                
    precision = TP/(TP+FP)  
    recall = TP/(TP+FN)
    #print(f"How well our model can classify binary outcomes: accuracy of {accuracy}, precision of {precision}, and recall of {recall}")
    print("How well our model can classify binary outcomes: accuracy of %.3f precision of %.3f, and recall of %.3f" % (accuracy, precision, recall))
    
    return accuracy, precision, recall
