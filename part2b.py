#getting the accuracy using KNN classification automatically using sklearn library
# PS : change the location of the train and test files

import pickle
import ASS1.part1
from sklearn.neighbors import KNeighborsClassifier

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():

    # getting the data
    batch = []
    batch.append(unpickle('/home/alpha/Work/DLCV/Ass_1/cifar-10-python/cifar-10-batches-py/data_batch_1'))
    batch.append(unpickle('/home/alpha/Work/DLCV/Ass_1/cifar-10-python/cifar-10-batches-py/data_batch_2'))
    batch.append(unpickle('/home/alpha/Work/DLCV/Ass_1/cifar-10-python/cifar-10-batches-py/data_batch_3'))
    batch.append(unpickle('/home/alpha/Work/DLCV/Ass_1/cifar-10-python/cifar-10-batches-py/data_batch_4'))
    batch.append(unpickle('/home/alpha/Work/DLCV/Ass_1/cifar-10-python/cifar-10-batches-py/data_batch_5'))

    train_features,train_labels=ASS1.part1.get_train_features(batch)

    test_batch=unpickle('/home/alpha/Work/DLCV/Ass_1/cifar-10-python/cifar-10-batches-py/test_batch')

    test_features,test_labels=ASS1.part1.get_test_features(test_batch)

    accuracy={}

    for k in range(3,12,2):
        neigh=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
        neigh.fit(train_features,train_labels)
        acc=neigh.score(test_features,test_labels)
        accuracy[str(k)+'nearest-neighbor']=acc

    #print(accuracy) #uncomment this to print the accuracy

if __name__=='__main__':
    main()