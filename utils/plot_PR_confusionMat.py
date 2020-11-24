#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:50:27 2020

@author: sharib
"""

###############################################################################
# plots below: 1) PR curve 2) Confusion matrix
###############################################################################
# The average precision score in multi-label settings
# ....................................................
# n_classes = 4
# n_classes = 3
n_classes = 2

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# For each class
def confusionPlot (y_test1, predicted, resultFileName, classA, classB):
     # plot confusion matrix
    import seaborn as sns
    import pandas as pd
    fontsize=20
    figsize = (10,7)
    fig = plt.figure(figsize=figsize)
    cf_AAM = metrics.confusion_matrix(y_test1, predicted)
    
    # for 3 way there are only 3 classes
    if n_classes == 4:
        df_cm = pd.DataFrame(cf_AAM, index=['Squamous', 'Barretts', 'Dysplasia', 'Cancer'], columns=['Squamous', 'Barretts', 'Dysplasia', 'Cancer'], )
    elif (n_classes == 3):
        df_cm = pd.DataFrame(cf_AAM, index=['Squamous', 'Barretts', 'Dysplasia'], columns=['Squamous', 'Barretts', 'Dysplasia'], )
    else:
        if classA == '1' and  classB == '2':
            df_cm = pd.DataFrame(cf_AAM, index=['Squamous', 'Barretts'], columns=['Squamous', 'Barretts'], )
        elif classA == '1' and  classB == '3':
            df_cm = pd.DataFrame(cf_AAM, index=['Squamous', 'Dysplasia'], columns=['Squamous', 'Dysplasia'], )
        else:
            df_cm = pd.DataFrame(cf_AAM, index=['Barretts', 'Dysplasia'], columns=['Barretts', 'Dysplasia'], )
        
    k_way = 0
    if k_way==3:  
        heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 24,}, fmt="d", cmap="Reds")
    else:
        heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 24,}, fmt="d", cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize,fontweight='bold' )
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize, fontweight='bold')
    fig.savefig(resultFileName, dpi=300)
    return cf_AAM
    
    
def plot_PR_curve(y_test1, y_score, predicted, resultFileName):
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    # TODO: add 0 and 1 to the corresponding class label
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test1)
    Y_test = lb.transform(y_test1)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
    ###############################################################################
    # Plot the micro-averaged Precision-Recall curve
    # ...............................................
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    plt.show()
    fig.savefig('avgPrec_recall'+ resultFileName, dpi=300)
    ###############################################################################
    # Plot Precision-Recall curve for each class and iso-f1 curves
    # .............................................................
    #
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for Spectra from MuSE study (multi-class)')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()
    fig.savefig(resultFileName, dpi=300)

    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
