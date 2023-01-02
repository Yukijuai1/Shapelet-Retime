from __future__ import division, print_function

from os.path import expanduser

from sklearn.metrics import classification_report,accuracy_score

from shapelets_lts.classification import LtsShapeletClassifier
from shapelets_lts.util import ucr_dataset_loader, plot_sample_shapelets
from pyts.classification import LearningShapelets
import numpy as np
import pandas as pd
import argparse
from pyts.transformation import ShapeletTransform

"""
This example uses dataset from the UCR archive "UCR Time Series Classification
Archive" format.  

- Follow the instruction on the UCR page 
(http://www.cs.ucr.edu/~eamonn/time_series_data/) to download the dataset. You 
need to be patient! :) 
- Update the vars below to point to the correct dataset location in your  
machine.

Otherwise update _load_train_test_datasets() below to return your own dataset.
"""

ucr_dataset_base_folder = expanduser('~/ws/data/UCR_TS_Archive_2015/')
ucr_dataset_name = 'Gun_Point'

train_filepath='/home/ubuntu/xuexi/2022_2/nn/final/main/data/data_simple/train2.txt'
test_filepath='/home/ubuntu/xuexi/2022_2/nn/final/main/data/data_simple/test2.txt'

def read_datasets(train_filepath,test_filepath):
    #train_df=pd.DataFrame(pd.read_csv(train_filepath, sep='\t'))
    #test_df=pd.DataFrame(pd.read_csv(test_filepath, sep='\t'))
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
def main():
    args = parse_arguments()
    train_filepath='UCRArchive_2018/'+str(args.dataset)+'/'+str(args.dataset)+'_TRAIN.tsv'
    test_filepath='UCRArchive_2018/'+str(args.dataset)+'/'+str(args.dataset)+'_TEST.tsv'
    # load the data
    print('\nLoading data...')
    x_train, y_train, x_test, y_test = read_datasets(train_filepath,test_filepath)

    
    # create a classifier
    Q = x_train.shape[1]
    K = int(0.15 * Q)
    L_min = int(0.2 * Q)
    '''
    clf = LtsShapeletClassifier(
        K=K,
        R=3,
        L_min=L_min,
        epocs=30,
        lamda=0.01,
        eta=0.01,
        shapelet_initialization='segments_centroids',
        plot_loss=True
    )
    '''
    clf = LearningShapelets()

    # train the classifier
    print('\nTraining...')
    clf.fit(x_train, y_train)

    # evaluate on test data
    print('\nEvaluating...')
    y_pred = clf.predict(x_test)
    print('accuracy ',accuracy_score(y_pred, y_test))
    print(
        'classification report...\n{}'
        ''.format(classification_report(y_true=y_test, y_pred=y_pred))
    )

    # plot sample shapelets
    #print('\nPlotting sample shapelets...')
    #plot_sample_shapelets(shapelets=clf.get_shapelets(), sample_size=36)
    


if __name__ == '__main__':
    main()
