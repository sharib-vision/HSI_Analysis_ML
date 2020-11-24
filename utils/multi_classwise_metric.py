#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:59:13 2020

@author: sharib
"""


from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
 
def compute_cfMatrix(y_pred, y_true, labels):
    sensitivity = []
    specificity = []
    PPV = []
    NPV = []
    Accuracy = []
    F1 = []
    
    perClass_cfM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    epsilon = 0.00001
    for i in range (len(labels)):
        TN = perClass_cfM[i][0,0]
        FN = perClass_cfM[i][1,0]
        TP = perClass_cfM[i][1,1]
        FP = perClass_cfM[i][0,1]
        # recall
        r = TP/(TP+FN+epsilon)
        sensitivity.append(r)
        # specificity
        specificity.append(TN/(TN+FP+epsilon))
        # precision
        p = TP/(TP+FP+epsilon)
        PPV.append(p)
        # F1-score
        F1.append(2*(p*r)/(p+r+epsilon))
        # NPV
        NPV.append(TN/(TN+FN+epsilon))
        
        Accuracy.append((TP+TN+epsilon)/(TP+TN+FP+FN+epsilon))

    return sensitivity, specificity, PPV, NPV, Accuracy, F1



def compute_metricswithCFMatrix(confusion_matrix):
    sensitivity = []
    specificity = []
    PPV = []
    NPV = []
    Accuracy = []
    F1 = []    

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP/(TP+FN)
    # Specificity or true negative rate
    specificity = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    
    F1 = 2*TP/(2*TP+FP+FN)
    
    # Overall accuracy
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    
    return sensitivity, specificity, PPV, NPV, Accuracy,FPR, FNR, FDR, F1


def saveMetrics2Json( method, sensitivity, specificity, PPV, NPV, Accuracy,FPR, FNR, FDR, F1):
    my_dictionary = { "HSI":{
            "method":{
            "value":  method
            },
            "sensitivity":{
            "value": sensitivity.tolist()
            },
            "specificity":{
            "value": specificity.tolist()
            },
            "PPV":{
            "value": PPV.tolist(),
            },
            "NPV":{
            "value": NPV.tolist(),
            }, 
            "Accuracy":{
            "value": Accuracy.tolist(),
            },
            "FPR":{
            "value": FPR.tolist(),
            },
            "FNR":{
            "value": FNR.tolist(),
            },
            "FDR":{
            "value": FDR.tolist(),
            },
            "F1":{
            "value": F1.tolist(),
            },                   
            "meanAccuracy":{
            "value": np.mean(Accuracy),
            },             
            }
        }   
    return my_dictionary
