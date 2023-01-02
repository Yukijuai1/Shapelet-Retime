import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse


def read_datasets(train_filepath,test_filepath):
    train_df=pd.DataFrame(np.loadtxt(train_filepath))
    test_df=pd.DataFrame(np.loadtxt(test_filepath))
    X, y = train_df[train_df.columns[1:]].values,train_df[train_df.columns[0]].values.astype(int)
    time_series, labels = test_df[test_df.columns[1:]].values,test_df[test_df.columns[0]].values
    return X,y,time_series,labels

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Choose DataSet'
    )
    parser.add_argument('--dataset', type=str,  required=True,
                        help='activation function')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    train_filepath='UCRArchive_2018/'+str(args.dataset)+'/'+str(args.dataset)+'_TRAIN.tsv'
    test_filepath='UCRArchive_2018/'+str(args.dataset)+'/'+str(args.dataset)+'_TEST.tsv'
   # load the data
    print('Loading data...')
    x_train, y_train, x_test, y_test = read_datasets(train_filepath,test_filepath)
    x_train=x_train
    y_train=y_train
    x_test=x_test
    y_test=y_test

    # create a classifier
    print('\nBuilding classifier...')
    clf=KNeighborsClassifier(n_neighbors=10)


    # train the classifier
    print('\nTraining...')
    clf.fit(x_train, y_train)

    # evaluate on test data
    print('\nPredicting...')
    time_1 = time.time()
    predicted = clf.predict(x_test)
    print('accuracy ',accuracy_score(predicted, y_test))
    report = classification_report(y_test,predicted)
    print(report)
    time_2 = time.time()
    print('cost ', time_2 - time_1, ' second', '\n')
