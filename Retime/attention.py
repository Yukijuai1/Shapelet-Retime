import torch
from torch import nn, einsum
from einops import rearrange
from torch.nn import Linear


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, d_k=64, dropout=0.1) -> None:
        super().__init__()
        if not self.ifExist(context_dim):
            context_dim = query_dim
        self.heads = heads
        self.scale = d_k ** -0.5
        d_model = d_k * heads
        self.to_q = nn.Linear(query_dim, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * d_model, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, query_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        if not self.ifExist(context):
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda mat: rearrange(mat, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        qkT = einsum('b n d, b m d->b n m', q, k) * self.scale
        attention = qkT.softmax(dim=-1)
        attention = einsum('b n m, b m d->b n d', attention, v)
        attention = rearrange(attention, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(attention)

    @staticmethod
    def ifExist(var):
        if var is None:
            return False
        else:
            return True

def contentAttention(input):
    batch_size=input.size()[0]
    length=input.size()[1]
    var=input.size()[2]
    atten=MultiHeadAttention(query_dim=var)
    key=[]
    for i in range(length):
        index=torch.tensor([i])
        key.append(torch.index_select(input,1,index).view(batch_size,var))
    tmp=torch.cat([key[0], key[1]], 0)
    for i in range(2,length):
        tmp=torch.cat([tmp, key[i]], 0)
    tmp=tmp.view(length,batch_size,var)
    tmp=atten(tmp)
    key1=[]
    for i in range(batch_size):
        index=torch.tensor([i])
        key1.append(torch.index_select(tmp,1,index).view(length,var))
    res=torch.cat([key1[0], key1[1]], 0)
    for i in range(2,batch_size):
        res=torch.cat([res, key1[i]], 0)
    res=res.view(batch_size,length,var)
    return res

def TemporalAttention(input):
    batch_size=input.size()[0]
    length=input.size()[1]
    var=input.size()[2]
    atten=MultiHeadAttention(query_dim=var)
    return atten(input)

class ForWard(nn.Module):
    def __init__(self,T,d):
        super(ForWard, self).__init__()

        self.linear1 = Linear(T, d)
        self.linear2 = Linear(d, T)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        return output

if __name__ == "__main__":
    
    ## 类实例化
    attention = MultiHeadAttention(query_dim=20)
    
    ## 输入
    qurry = torch.randn(129, 12, 20)
    key=[]
    for i in range(12):
        index=torch.tensor([i])
        key.append(torch.index_select(qurry,1,index).view(129,20))
    tmp=torch.cat([key[0], key[1]], 0)
    for i in range(2,12):
        tmp=torch.cat([tmp, key[i]], 0)
    tmp=tmp.view(12,129,20)
    tmp=attention(tmp)
    key1=[]
    for i in range(129):
        index=torch.tensor([i])
        key1.append(torch.index_select(tmp,1,index).view(12,20))
    res=torch.cat([key1[0], key1[1]], 0)
    for i in range(2,129):
        res=torch.cat([res, key1[i]], 0)
    res=res.view(129,12,20)
    print(res[5,1],tmp[1][5])
    