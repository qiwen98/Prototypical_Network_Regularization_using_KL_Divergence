
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import repeat
import pandas as pd



def clip_dist(query_embeddings,prototype_embeddings):
    """
    Ref from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
    Args:
    - query:  # shape (50,1028)
    - prototype: (10,1028) -< can expand to (50,1028) using protyte.expand(50,1028)
    """
    temperature = 1.0
    # Calculating the Loss
    logits = (query_embeddings @ prototype_embeddings.T) / temperature # (50,50)
    prototype_similarity = prototype_embeddings @ prototype_embeddings.T # (50,50)
    query_similarity = query_embeddings @ query_embeddings.T # (50,50)
    targets = F.softmax(
        (prototype_similarity + query_similarity) / 2 * temperature, dim=-1
    )
    query_loss = cross_entropy(logits, targets, reduction='none')
    prototype_loss = cross_entropy(logits.T, targets.T, reduction='none')
    dist =  (prototype_loss + query_loss) / 2.0 # shape: (batch_size)
    return dist #!!!!!!!!! need find some way to map it to (50,10) 

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def kl_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)



    # torch.pow(x - y, 2).sum(2)

    return torch.nn.functional.kl_div(x,y)



if __name__ == '__main__':
    a = torch.rand(50,1028)
    b = torch.rand(10,1028)
    # a = F.log_softmax(a,dim=1)
    # print(a)
    # # b = F.log_softmax(b,dim=1)
    # dist = kl_dist(a,b)
    # print(dist)
    # print( F.log_softmax(-dist, dim=1))
    # print( F.log_softmax(-dist, dim=1).max())
    # print( F.log_softmax(-dist, dim=1).min())
    # print(dist.view(10,5,10))
    # print(dist.shape)
    n_classes=10
    n_query=5
    y_pred = torch.Tensor([[0, 0, 0, 0, 0],
        [6, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 2, 5, 5, 5],
        [6, 3, 6, 3, 6],
        [7, 7, 7, 7, 3],
        [8, 8, 8, 8, 0],
        [1, 9, 9, 6, 9]])
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1)
    target_inds = target_inds.expand(n_classes, n_query).float()# 10 class, for each class 5 samples 

    
    print(target_inds)
    # x = F.log_softmax(target_inds)
    # y = F.log_softmax(y_pred)
    # print(x)
    # print(y)

    # print(torch.nn.functional.kl_div(x,y,log_target=True))

    x = y_pred.numpy().flatten()
    y = target_inds.numpy().flatten()

    Methods=[]
    Value = []
    Methods.extend(repeat('Y_Pred',50))
    Methods.extend(repeat('Y_Target',50))

    Value.extend(x)
    Value.extend(y)




    plot_dic=defaultdict(list)
    plot_dic['Distribution']=Methods
    plot_dic['10 way(Class) Bins']=Value
    # x  = np.random.normal(size=500)
    # x = x[:, 0]
    # y = x[:, 1]


    df = pd.DataFrame.from_dict(plot_dic)




    # plt.hist(x, bins=10)
    sns.set_theme()
    sns.displot(df,x='10 way(Class) Bins',hue = "Distribution", kind="kde",cut=0).set(title='KL Regularization during Training')
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')