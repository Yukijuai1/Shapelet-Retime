import torch
from torch import nn
from torchsummary import summary
import torch.utils.data as Data
from mlp import MLP
from input import Linearlayer,PositionalEncoding
from attention import MultiHeadAttention,contentAttention,TemporalAttention,ForWard
from shapelet import find_shapelets_bf
import numpy as np



#InputModule(a,b,12,1,20)
def InputModule(X,Y,T,v,d):
    linear=Linearlayer(v,d)
    pe=PositionalEncoding(d_model=d)
    X_norm = nn.LayerNorm([X.size(0),T,d])
    Y_norm = nn.LayerNorm([Y.size(0),T,d])
    X=linear(X)
    X=pe(X)
    X=X_norm(X)
    Y=linear(Y)
    Y=pe(Y)
    Y=Y_norm(Y)
    return torch.cat([X,Y],0)


def AggregationModule(input,T,d,L):
    out =input
    for i in range(L):
        out=contentAttention(input)
        out=TemporalAttention(input)
        forward=ForWard(d,50)
        out=forward(out)
    return out



def OutputModule(X_train,y_train,X_test,d,epochs):
    #print(X_train,y_train,X_test)
    net=MLP(d,100,50)
    #summary(net, input_size=(1, 12,20))
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)
    loss_func = nn.MSELoss()
    train_loss_all = []
    for epoch in range(epochs):
        train_loss = 0
        train_num = 0
        output = net(X_train) # MLP在训练batch上的输出
        loss = loss_func(output[:X_train.size()[0]-1], y_train) # 均方根损失函数
        optimizer.zero_grad() # 每次迭代梯度初始化0
        loss.backward(retain_graph=True) # 反向传播，计算梯度
        optimizer.step() # 使用梯度进行优化
        train_loss += loss.item() * X_train.size(0)
        train_num += X_train.size(0)
        train_loss_all.append(train_loss / train_num)
    
    return net(X_test[0]),train_loss_all

def PretrianModule(X_train,y_train):
    from sklearn.cluster import KMeans
    y_pred = KMeans(n_clusters=2).fit_predict(X_train)
    X_pos,y_pos=[],[]
    X_neg,y_neg=[],[]
    pos,neg=[],[]
    for i in range(len(X_train)):
        if int(y_pred[i])==0:
            X_pos.append(X_train[i])
            y_pos.append(y_train[i])
            pos.append((X_train[i],y_train[i]))
        elif int(y_pred[i])==1:
            X_neg.append(X_train[i])
            y_neg.append(y_train[i])
            neg.append((X_train[i],y_train[i]))

    shapelet_pos, dist_pos=find_shapelets_bf(pos, max_len=len(X_train[0])-1, min_len=len(X_train[0])-1)
    shapelet_neg, dist_neg=find_shapelets_bf(neg, max_len=len(X_train[0])-1, min_len=len(X_train[0])-1)
    return np.array(X_pos),np.array(y_pos),np.array(X_neg),np.array(y_neg),shapelet_pos,shapelet_neg

if __name__ == "__main__":


    batch_size=20
    length=12
    var_num=1
    hidden_len=50
    epoch=5


    a=torch.randn(1,length,var_num)
    b=torch.randn(batch_size,length,var_num)
    labels=torch.ones(batch_size+1,length)


    input=InputModule(a,b,length,var_num,hidden_len)


    aggregate=AggregationModule(input,length,hidden_len,epoch)
    #print(aggregate.size())
    #print(aggregate[-1,:].unsqueeze(0))

    out,loss=OutputModule(aggregate,labels,aggregate[-1,:].unsqueeze(0),d=hidden_len,epochs=epoch)
    print(torch.max(out, 1))






