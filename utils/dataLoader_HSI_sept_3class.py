#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:39:33 2020

@author: sharib
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# standardize data
def scale_data(trainX, standardize):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:]
	# flatten windows
	longX = longX.reshape((longX.shape[0] , longX.shape[1]))
	# flatten train and test
	flatTrainX = trainX.reshape((trainX.shape[0] , trainX.shape[1]))
	# standardize
	if standardize:
		s = StandardScaler()
		# fit on training data
		s.fit(flatTrainX)
		flatTrainX = s.transform(flatTrainX)
	# reshape
	flatTrainX = flatTrainX.reshape((trainX.shape))
	return flatTrainX

class MUSE_HSI_dataLoader_noFile(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # in csv file put one below
        # data = pd.read_csv(csv_file_data, header = None)#dont do header = None here otherwise will take 0 1 2 ...
        # df = pd.DataFrame(data.values)
        
        # X = np.array(df, dtype=float)
        
        # standarize data for both train and test
        # X = scale_data(X, standardize=True)

        self.data = X
        # ytrain = pd.read_csv(csv_file_labels, header=None)
        
        # # # perform pca
        pca = PCA()
        X = pca.fit_transform(X)

        # y_en= np.array(ytrain[0])

        # y = y_en.reshape(ytrain.shape[0],1)
        # uncomment if origina lables with patient id is used
        # y = y_en[:, 1].reshape(ytrain.shape[0],1)
        
        self.labels =  y
        
        self.transform = transform
        # mu, sigma = np.mean(X), np.std(X)
        # mu, sigma = 0, 1
        # self.noise= np.random.normal(mu, sigma, [X.shape[0], X.shape[1]]) 
            

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        featureArray =  self.data[idx]
        
            
        # comment this for 2 class classification     
        label = self.labels[idx][0]-1
        #-1
        # if self.transform:
        #     featureArray = self.data[idx] + self.noise[idx]
    
        # sample = {'feature': (torch.tensor(featureArray).type(torch.float)).unsqueeze(0), 'label': torch.tensor(label) }
        sample = {'feature': (torch.tensor(featureArray).type(torch.float)).unsqueeze(0), 'label': torch.tensor(label) }

        # mu, sigma = 0, 0.1 
        # noise = np.random.normal(mu, sigma, [X_train1.shape[0],X_train1.shape[1]]) 
        # creating a noise with the same dimension as the dataset (2,2) 

    

        return sample


class MUSE_HSI_dataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_data, csv_file_labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # in csv file put one below
        data = pd.read_csv(csv_file_data, header = None)#dont do header = None here otherwise will take 0 1 2 ...
        df = pd.DataFrame(data.values)
        
        X = np.array(df, dtype=float)
        
        # standarize data for both train and test
        # X = scale_data(X, standardize=True)

        self.data = X
        ytrain = pd.read_csv(csv_file_labels, header=None)
        
        # # perform pca
        # pca = PCA()
        # X = pca.fit_transform(X)

        y_en= np.array(ytrain[0])

        y = y_en.reshape(ytrain.shape[0],1)
        # uncomment if origina lables with patient id is used
        # y = y_en[:, 1].reshape(ytrain.shape[0],1)
        
        self.labels =  y
        
        self.transform = transform
        
        # change here if classmax is 3
        self.classmax = 3
        # mu, sigma = np.mean(X), np.std(X)
        # mu, sigma = 0, 1
        # self.noise= np.random.normal(mu, sigma, [X.shape[0], X.shape[1]]) 
            

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        featureArray =  self.data[idx]
        
        # uncomment this for 2 class classification     
        # if self.classmax == 2:
        #     label = self.labels[idx][0]-1
        # else:
        #     if self.labels[idx][0] == 3:
        #         label = self.labels[idx][0]-2
        #     if (self.labels[idx][0] == 1):
        #         label = self.labels[idx][0]-1
        #     if  (self.labels[idx][0] == 2) :
        #         label = self.labels[idx][0]-2 # 2 for [2,3] , 1 else

            
            
        # comment this for 2 class classification     
        label = self.labels[idx][0]-1
        #-1
        # if self.transform:
        #     featureArray = self.data[idx] + self.noise[idx]
    
        # sample = {'feature': (torch.tensor(featureArray).type(torch.float)).unsqueeze(0), 'label': torch.tensor(label) }
        sample = {'feature': (torch.tensor(featureArray).type(torch.float)).unsqueeze(0), 'label': torch.tensor(label) }

        # mu, sigma = 0, 0.1 
        # noise = np.random.normal(mu, sigma, [X_train1.shape[0],X_train1.shape[1]]) 
        # creating a noise with the same dimension as the dataset (2,2) 

    

        return sample
    
# plot a histogram of each variable in the dataset
def plot_variable_distributions(trainX):
    from matplotlib import pyplot
    cut = int(trainX.shape[1] / 2)
    print('cut:', cut)
    longX = trainX[:, -cut:, :]
	# flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    print(longX.shape)
    pyplot.figure()
    xaxis = None
    for i in range(longX.shape[1]):
        ax= pyplot.subplot(longX.shape[1], 1, i+1, sharex=xaxis)
        ax.set_xlim(-1, 1)
        if i == 0:
            xaxis = ax
        pyplot.hist(longX[:, i], bins=100)
    pyplot.show()
            
 
if __name__ == '__main__':

    # hsi_dataset = MUSE_HSI_dataLoader(csv_file_data='data_cam/train_processed_cleandata.csv',
    #                                 csv_file_labels='data_cam/train_processed_cleanlabel.csv')
    
    
    hsi_dataset = MUSE_HSI_dataLoader(csv_file_data='data_cam/train_processed_data.csv',
                                    csv_file_labels='data_cam/train_processed_label.csv')
    
    
    dataloader = DataLoader(hsi_dataset, batch_size=4, shuffle=True, num_workers=4) 
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['feature'].size(),
              sample_batched['label'].size())
        
        
    
    