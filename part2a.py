#getting the accuracy using KNN classification manually using chi-squared similarity martix
# PS : change the location of the train and test files

import pickle
import numpy as np
import cv2
import operator
import ASS1.part1

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# take first element for sort
def takeFirst(elem):
    return elem[0]

def main():

    # getting the data
    batch = []
    batch.append(unpickle('G:\study material\DLFCV\programming assignments\\ass1\cifar-10-batches-py\data_batch_1'))
    batch.append(unpickle('G:\study material\DLFCV\programming assignments\\ass1\cifar-10-batches-py\data_batch_2'))
    batch.append(unpickle('G:\study material\DLFCV\programming assignments\\ass1\cifar-10-batches-py\data_batch_3'))
    batch.append(unpickle('G:\study material\DLFCV\programming assignments\\ass1\cifar-10-batches-py\data_batch_4'))
    batch.append(unpickle('G:\study material\DLFCV\programming assignments\\ass1\cifar-10-batches-py\data_batch_5'))

    train_features,train_labels=ASS1.part1.get_train_features(batch)

    test_batch=unpickle('G:\study material\DLFCV\programming assignments\\ass1\cifar-10-batches-py\\test_batch')

    test_features,test_labels=ASS1.part1.get_test_features(test_batch)
    correctly_deduced=0
    incorrectly_deduced=0

    #do knn classification
    accuracy=[]

    for n in range(3,12,2):

        for k in range(0,np.shape(test_features)[0]):
            result=[]
            # compute the distance between the two histograms
            # using the chi-squared similarity method
            for i in range(0,np.shape(train_features)[0]):
                #hist.convertTo(hist,cv2.CV_32F)
                #train_features[i][j].convertTo(train_features[i][j],cv2.CV_32F)
                d=cv2.compareHist(test_features[0][k],train_features[i][j], cv2.HISTCMP_CHISQR)
                result.append([d,train_labels[i]]) #appending the similarity value and class of the training sample it got comapred to

            #get the top 3 most similar values
            result=sorted(result,key=takeFirst,reverse=True)[0:n]
            count_classes={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

            for i in range(0,n):
                count_classes[result[i][1]]+=1

            #get the maximum majority class
            class_for_sample=max(count_classes.items(), key=operator.itemgetter(1))[0]

            if class_for_sample==test_batch[b'labels'][k]:
                correctly_deduced+=1
            else:
                incorrectly_deduced+=1

        accuracy.append((correctly_deduced/(correctly_deduced+incorrectly_deduced))*100)

if __name__=='__main__':
    main()
