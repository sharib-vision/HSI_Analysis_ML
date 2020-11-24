#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:16:57 2020

 
"""

from torch.utils.data.sampler import SubsetRandomSampler
import torch 
import argparse


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def get_argparser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default='./data/noExclusion_test_data.csv',
                        help="path to train")
    parser.add_argument("--test_label", type=str, default='./data/noExclusion_test_label.csv',
                        help="path to train")
    
    parser.add_argument("--fileType", type=str, default='noExclusion',
                        help="'noExclusion', 'withExclusion', 'balanced'")
    
    parser.add_argument("--num_labels", type=int, default=3,
                        help="num classes (default: 2/3 class)")

    parser.add_argument("--ckptFolderAndFile", type=str, default='./ckpt/noExclusion_train_data_sept_nclasses_3_v20.pth',
                        help="checkpoint directory")
    
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    parser.add_argument("--classSplit", type=str, default='1,2,3', help="1,2")
    
    
    return parser

if __name__ == '__main__':
    

    import numpy as np
    from torch.utils.data import DataLoader
    from model.model import model
    import os
    import time
    import pandas as pd


    opts = get_argparser().parse_args()

    fileType = opts.fileType
    nlabel= opts.num_labels
    classSplit = opts.classSplit
    dataFile_test = opts.test_data
    labelFile_test = opts.test_label
    
    if nlabel == 2:
        from  utils.dataLoader_HSI_2class import MUSE_HSI_dataLoader
        from utils.plot_PR_confusionMat import confusionPlot 
        
        
        classA = int(classSplit.split(',')[0])
        classB = int(classSplit.split(',')[1])
        hsi_dataset_test = MUSE_HSI_dataLoader(csv_file_data=dataFile_test,
                                csv_file_labels=labelFile_test, class_categories=classSplit)
       
    else:
        from utils.dataLoader_HSI_sept_3class import MUSE_HSI_dataLoader
        from utils.plot_PR_confusionMat_3class import confusionPlot 
        hsi_dataset_test = MUSE_HSI_dataLoader(csv_file_data=dataFile_test,
                                csv_file_labels=labelFile_test)

        
    
    fileName = ((dataFile_test).split('/')[-1]).split('.')[0]
    
    directoryName = 'results_'+ fileName
    os.makedirs(directoryName, exist_ok = True)
      
    device = 'cuda'
    m1 = model(nlabel).to(device)

    ckptFile = opts.ckptFolderAndFile
    
    m1.load_state_dict(torch.load(ckptFile))

    
    dataloade_val = DataLoader(hsi_dataset_test, batch_size=1, num_workers=0) 

    m1.eval()
    ypred = []
    ylabel = []
    logps_= []
    st = time.time()    

    with torch.no_grad():
        for k, sample_val in enumerate(dataloade_val):
            inputs = torch.cuda.FloatTensor(sample_val['feature'].to(device))
            labels = torch.cuda.FloatTensor(sample_val['label'].to(device).type(torch.float).unsqueeze(1))
            
            logps = m1.predict(inputs) 
            logps_.append(logps.detach().to('cpu').numpy()[0])
            _, preds = torch.max(logps.data, 1)
            ypred.append(preds.detach().to('cpu').numpy()[0])
            ylabel.append(int(labels[0].detach().to('cpu').numpy()[0]))
            
    print('Elapsed test time: {}'.format(time.time()-st))    
     
    # get the accuracy/confusion matrix
    yl = np.array(ylabel)
    yp = np.array(ypred)

    #
    if  nlabel == 2:
        cf_Mat_CNN = confusionPlot(yl, yp, directoryName+'/'+fileName +'.svg', str(classA), str(classB) )
    else:
        cf_Mat_CNN = confusionPlot(yl, yp, directoryName+'/'+fileName + '_' +'3class.svg' )
    
    from sklearn import metrics
    print("classification result for :\n%s\n"
      % (metrics.classification_report(yl, yp)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(yl, yp))
    
    # write to csv file
    df = pd.DataFrame( ylabel )
    df.to_csv(os.path.join('./', directoryName, 'test_labels' + fileType + '.csv'))
    
    df = pd.DataFrame( logps_ )
    df.to_csv(os.path.join('./', directoryName, 'predicted_labels' + fileType + '.csv'))
    
    '''
    save all metric computations in a json file
    '''
    
    import json
    from utils.multi_classwise_metric import saveMetrics2Json, compute_metricswithCFMatrix

    jsonFileName = fileType+'_'+str(nlabel)+'_class.json'
    resultDIR = './metric_outputs/'
    os.makedirs(resultDIR, exist_ok=True)
    jsonFileName=os.path.join(resultDIR, jsonFileName)
    
    sensitivity, specificity, PPV, NPV, Accuracy,FPR, FNR, FDR, F1 = compute_metricswithCFMatrix(cf_Mat_CNN)
    if  nlabel == 2:
        my_dictionary = saveMetrics2Json(fileType+'_CNN_'+str(classA)+'_'+str(classB),sensitivity, specificity, PPV, NPV, Accuracy,FPR, FNR, FDR, F1)
    else: 
        my_dictionary = saveMetrics2Json(fileType+'_CNN_3classes',sensitivity, specificity, PPV, NPV, Accuracy,FPR, FNR, FDR, F1)
        
    fileObj= open(jsonFileName, "a")
    fileObj.write("\n")
    json.dump(my_dictionary, fileObj)
    
    fileObj.close()
    print('Evaluation metric values written to---->', (jsonFileName))