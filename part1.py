#getting the features(histogram and HOG of each image in all 5 training sets) manually from the dataset

import pickle
import numpy as np
import cv2
from skimage.feature import hog

def get_test_features(batch):

    #list containing histogram as feature for all the images in all the 5 training sets
    features=[]
    labels=[]

    for n in range(0,np.size(batch)): #for all the 5 training sets in batch
        for row in range(0,np.shape(batch[b'data'])[0]): #for all the images in 1 training set
            red = np.reshape(batch[b'data'][row][0:1024], (32, 32))
            blue = np.reshape(batch[b'data'][row][1024:2048], (32, 32))
            green = np.reshape(batch[b'data'][row][2048:3072], (32, 32))
            img=cv2.merge((blue,green,red))
           
            #using histogram as feature
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist,dst=None,dtype=cv2.CV_32F).flatten()            

            #using HOG as feature)
            hog_img=hog(img, feature_vector=True, multichannel=True)
            
            features.append(hist) # put hog_img instead of hist here to use HOG as a feature

            labels.append(batch[b'labels'][row])

    return features,labels

def get_train_features(batch):

    #list containing histogram as feature for all the images in all the 5 training sets
    features=[]
    labels=[]

    for n in range(0,np.size(batch)): #for all the 5 training sets in batch
        for row in range(0,np.shape(batch[n][b'data'])[0]): #for all the images in 1 training set
            red = np.reshape(batch[n][b'data'][row][0:1024], (32, 32))
            blue = np.reshape(batch[n][b'data'][row][1024:2048], (32, 32))
            green = np.reshape(batch[n][b'data'][row][2048:3072], (32, 32))
            img=cv2.merge((blue,green,red))
           
            #using histogram as a feature
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist,dst=None,dtype=cv2.CV_32F).flatten()

            #using HOG as a feature
            hog_img=hog(img,feature_vector=True,multichannel=True)

            features.append(hist)  # put hog_img instead of hist here to use HOG as a feature

            labels.append(batch[n][b'labels'][row])

    return features,labels