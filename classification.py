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
        
    scoreKNN = np.mean(predictions)
        
    if scoreKNN >= 0.5:
        new_prediction = 1
    else:
        new_prediction = 0
        
    return scoreKNN, new_prediction



def get_accuracy(y_results, y_te, score):
    """Checks whether prediction are accurate by compraing with y_te
    
    Args: 
        predictions:
        y_te:
    
    Returns:
        len(good_guess):
        len(bad_guess):
    """ 
    
    difference = (y_results-y_te)
    good_guess = difference[difference==0]
    bad_guess = difference[difference!=0]
    accuracy = len(good_guess)/(len(good_guess)+len(bad_guess))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(y_results.shape[0]):
        if difference[i] == 1:
            FP +=1
        if difference[i] == -1:
            FN +=1
        else :
            if y_results[i] == 1:
                TP +=1
            else:
                TN +=1       
                
    precision = TP/(TP+FP)  
    recall = TP/(TP+FN)
    auc = get_auc(score, y_te)
    #print(f"How well our model can classify binary outcomes: accuracy of {accuracy}, precision of {precision}, and recall of {recall}")
    print("How well our model can classify binary outcomes: accuracy of %.3f, precision of %.3f, recall of %.3f, and AUC score of %.3f" % (accuracy, precision, recall, auc))
    
    return accuracy, precision, recall



def get_auc(score, y_results):

    y = y_results

    # false positive rate
    FPR = []
    # true positive rate
    TPR = []
    # Iterate thresholds from 0.0 to 1.0
    thresholds = np.arange(0.0, 1.01, 0.001)
    print(len(thresholds))

    # get number of positive and negative examples in the dataset
    P = sum(y)
    N = len(y) - P

    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh in thresholds:
        FP=0
        TP=0
        thresh = round(thresh,2) 
        for i in range(len(score)):
            if (score[i] >= thresh):
                if y[i] == 1:
                    TP += 1
                if y[i] == 0:
                    FP += 1            
        FPR = np.append(FPR,FP/N)
        TPR = np.append(TPR, TP/P)

    #computing Arean Under Curve using the trapezoidal method
    auc = -1 * np.trapz(TPR, x=FPR)
    print(auc)

    
    plt.plot(FPR, TPR, marker='.', color='darkorange', label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label = 'No Discrimination')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('AUC_example.png')
    plt.show()
    
    return auc

