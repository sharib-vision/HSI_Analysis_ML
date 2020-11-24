#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:59:23 2020

@author: shariba
"""

from sklearn.metrics import fbeta_score, accuracy_score,confusion_matrix

def test_predict(clf, X_test, y_test): 
    results = {}
    predictions_test = clf.predict(X_test)
    
    accuracy=results['acc_test'] = accuracy_score(y_test, predictions_test)
    fbeta= results['f_test'] = fbeta_score(y_test, predictions_test,average='weighted', beta = 0.5)
    
    print("accuracy for {} model is {}".format(clf.__class__.__name__,results['acc_test']))
    print("F beta for {} model is {}".format(clf.__class__.__name__,results['f_test']))
    print('------------------------------------------------------------------')
    
    return results