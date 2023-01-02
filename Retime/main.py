import torch
from retime import InputModule,AggregationModule,OutputModule,PretrianModule
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import pickle
from shapelet import subsequence_dist
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

if __name__ == "__main__":

    args = parse_arguments()
    train_filepath='UCRArchive_2018/'+str(args.dataset)+'/'+str(args.dataset)+'_TRAIN.tsv'
    test_filepath='UCRArchive_2018/'+str(args.dataset)+'/'+str(args.dataset)+'_TEST.tsv'

    print('Loading data...')
    X_train, y_train, X_test, y_test = read_datasets(train_filepath,test_filepath)
    
    print('\nPreTraining...')
    X_pos,y_pos,X_neg,y_neg,shapelet_pos,shapelet_neg=PretrianModule(X_train, y_train)
    
    test_num=X_test.shape[0]
    X_test=torch.tensor(X_test, dtype=torch.float).view(X_test.shape[0],X_test.shape[1],1)

    
    print('\nTraining...')
    predict=[]
    for i in range(test_num):
        print('    Epoch: '+str(i))
        print('        Retrival...')
        if subsequence_dist(X_test[i],shapelet_pos) <= subsequence_dist(X_test[i],shapelet_pos):
            X_train_tmp=torch.tensor(X_pos, dtype=torch.float).view(X_pos.shape[0],X_pos.shape[1],1)
            y_train_tmp=torch.tensor(y_pos, dtype=torch.float)
        else:
            X_train_tmp=torch.tensor(X_neg, dtype=torch.float).view(X_neg.shape[0],X_neg.shape[1],1)
            y_train_tmp=torch.tensor(y_neg, dtype=torch.float)


        batch_size=X_train_tmp.size()[0]
        length=X_train_tmp.size()[1]
        var_num=X_train_tmp.size()[2]
        hidden_len=10
        epoch=1

        print('        InputModule...')
        input=InputModule(X_train_tmp,X_test[i].view(1,length,var_num),length,var_num,hidden_len)

        print('        AggregationModule...')
        aggregate=AggregationModule(input,length,hidden_len,epoch)
        aggregate=aggregate.view(batch_size+1,length*hidden_len)        
        aggregate=aggregate.detach().numpy()

        print('        MLPModule...')
        mlp = MLPClassifier(hidden_layer_sizes=(100,50), alpha=0.01, max_iter=10000)
        mlp.fit(aggregate[:batch_size],y_train_tmp)
        predict.append(int(mlp.predict(aggregate[-1,:].reshape(1,-1))))
        print('        Finish!')

    print('\nPredicting...')
    report = classification_report(y_test[:test_num],predict)
    print(report)
    #out,loss=OutputModule(aggregate,y_test,aggregate[-1,:].unsqueeze(0),d=length*hidden_len,epochs=epoch)
    
    