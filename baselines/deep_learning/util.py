import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from Model import Protonet,ResNet
from Datagenerator import Datagen_test
import numpy as np
from batch_sampler import EpisodicBatchSampler
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter1d, minimum_filter
from torch.autograd import Variable




def euclidean_dist(x, y):
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

    # 

    return torch.pow(x - y, 2).sum(2)

def clip_dist(query_embeddings,prototype_embeddings):
    """
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
    return loss.mean() #!!!!!!!!! need find some way to map it to (50,10) 

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()



def prototypical_loss(input, target,n_shot):
    '''
    Adopted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    Compute the prototypes by averaging the features of n_shot
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_shot: number of samples to keep in account when computing
      bprototypes, for each one of the current classes  # n_shot, n_samples 
    '''

    def supp_idxs(c):

        return target_cpu.eq(c).nonzero()[:n_shot].squeeze(1)

    target_cpu = target.to('cpu')
    # print('target',target)
    input_cpu = input.to('cpu')
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    p = n_classes * n_shot
    n_query = target.eq(classes[0].item()).sum().item() - n_shot
    support_idxs = list(map(supp_idxs,classes)) # we find all the index of the sample and arranged them into the 10 classes, eg: [ 2, 12, 22, 32, 42]-> first class, [ 1, 11, 21, 31, 41] -> second class
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) # find the middle embedding of of each cluster, we have 10 culster,each cluster with 5 samples/shots , (10, 1024) 10 middle point with 1024 dim

    query_idxs = torch.stack(list(map(lambda c:target.eq(c).nonzero()[n_shot:],classes))).view(-1)
    query_samples = input.cpu()[query_idxs]  # we have 50 query here, with 10 differnt classes  (50, 1024) 50 queires with 1024 dims

    dists = euclidean_dist(query_samples, prototypes)  # x ->(50,1024), y->(10,1024), distance -> (50,10)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) # apply log softmax to each class, the log probability of predicted y to each class, 
    # print(log_p_y.shape) # shape with 10,5,10 -> 10 classes , 5 shot and for each shot , we have the probability of each shot w.r.t each class (total 10)
    # print(log_p_y)
    
    # we want to make log_p_gt and log_p_y as close as possible  

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()# 10 class, for each class 5 samples 

    # print(-log_p_y.gather(2, target_inds))  # we gather the log_prob_y class by taking [:,:,0], [:,:,1], [:,:,2]......[:,:,9]
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() # idealy we want the value after mean approaching 0 as possible, we have 10 class here , in each class the embedding of the query should be 0 to the target class 
    # print(loss_val) 

    _, y_hat = log_p_y.max(2) # we get the predicted class index see example below,  # we have 50 query here, with 10 differnt classes 
    """
    [[0, 0, 0, 0, 0],
        [6, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 2, 5, 5, 5],
        [6, 3, 6, 3, 6],
        [7, 7, 7, 7, 3],
        [8, 8, 8, 8, 0],
        [1, 9, 9, 6, 9]
    """
    ## newly added kl divergence
    target_inds = target_inds.squeeze().float()
    y = F.log_softmax(target_inds)
    x = F.log_softmax(y_hat.float())

    loss_val = loss_val +torch.nn.functional.kl_div(x,y,log_target=True)
    ## newly added parts

    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


def get_probability(x_pos,neg_proto,query_set_out):


    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """

    pos_prototype = x_pos.mean(0)
    prototypes = torch.stack([pos_prototype,neg_proto]) #torch.Size([2, 1024]) 
    # print(prototypes.shape)
    # print(query_set_out.shape)
    dists = euclidean_dist(query_set_out,prototypes) # query set out torch.Size([8, 1024])
    # print(dists)
    '''  Taking inverse distance for converting distance to probabilities'''
    inverse_dist = torch.div(1.0, dists)
    # print(inverse_dist)
    prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0] # we just take the first one w.r.t the postive label we want to find

    """
    tensor([[0.4961, 0.5039], 
        [0.4966, 0.5034],
        [0.4937, 0.5063],
        [0.4947, 0.5053],
        [0.4956, 0.5044],
        [0.4960, 0.5040],
        [0.4967, 0.5033]]
    """

    

    return prob_pos.detach().cpu().tolist()



def evaluate_prototypes(conf=None,hdf_eval=None,device= None,strt_index_query=None):

    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel)

    gen_eval = Datagen_test(hdf_eval,conf)
    X_pos, X_neg,X_query = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)
    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False)
    query_set_feat = torch.zeros(0,1024).cpu()


    if conf.train.encoder == 'Resnet':
        Model  = ResNet()
    else:
        Model = Protonet()


    if device == 'cpu':
        Model.load_state_dict(torch.load(conf.path.best_model, map_location=torch.device('cpu')))
    else:
        Model.load_state_dict(torch.load(conf.path.best_model))

    Model.to(device)
    Model.eval()

    ##### change here
    'List for storing the combined probability across all iterations'
    prob_comb = []

    iterations = conf.eval.iterations
    for i in range(iterations):
        prob_pos_iter = []
        neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
        X_neg = X_neg[neg_indices]
        Y_neg = Y_neg[neg_indices]
        batch_size_neg = conf.eval.negative_set_batch_size
        neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
        negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None, batch_size=batch_size_neg)

        batch_samplr_pos = EpisodicBatchSampler(Y_pos, num_batch_query + 1, 1, conf.train.n_shot)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=batch_samplr_pos)

        neg_iterator = iter(negative_loader)
        pos_iterator = iter(pos_loader)
        q_iterator = iter(q_loader)

        print("Iteration number {}".format(i))

        for batch in tqdm(neg_iterator):
            x_neg, y_neg = batch
            x_neg = x_neg.to(device)
            feat_neg = Model(x_neg)
            feat_neg = feat_neg.detach().cpu()
            query_set_feat = torch.cat((query_set_feat, feat_neg), dim=0)
        neg_proto = query_set_feat.mean(dim=0)
        neg_proto =neg_proto.to(device)

        for batch in tqdm(q_iterator):
            x_q, y_q = batch
            x_q = x_q.to(device)
            x_pos, y_pos = next(pos_iterator)
            x_pos = x_pos.to(device)
            x_pos = Model(x_pos)
            x_query = Model(x_q)
            probability_pos = get_probability(x_pos, neg_proto, x_query)
            prob_pos_iter.extend(probability_pos)

        prob_comb.append(prob_pos_iter)
    prob_final = np.mean(np.array(prob_comb),axis=0)

    neg_iterator = iter(negative_loader)
    pos_iterator = iter(pos_loader)
    q_iterator = iter(q_loader)

    # print("----------models out------------")
    # # model outs
    # print(x_query.shape)
    # print(x_pos.shape)
    # print(feat_neg.shape)# x_neg
    # print(neg_proto.shape)

    # print("----------query ori iterator------------")
    # x_q, y_q = next(q_iterator)

    # print(x_q.shape)
    # print(y_q.shape) 

    # print("----------negative ori iterator------------")
    # x_neg, y_neg = next(neg_iterator)
    # print(x_neg.shape)
    # print(y_neg.shape)

    # # ori x_pos y_pos shape
    # print("----------positive ori iterator------------")
    # x_pos, y_pos = next(pos_iterator)

    # print(x_pos.shape)
    # print(y_pos.shape)


    ### change here 
    ### output the last layer of nueral netwotk where after applied sofmax (n*1) probablity of each n belongs to positive class


    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > 0.5, 1, 0)

    prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)


    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query

    offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset
