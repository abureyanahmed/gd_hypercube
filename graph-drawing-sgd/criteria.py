from pynndescent import NNDescent
from utils import lovasz_losses as L
from utils import utils

import torch
from torch import nn
from torch import optim
import numpy as np
import networkx as nx

import random



def crossings(pos, G, k2i, sampleSize, sampleOn='edges', reg_coef=1, niter=30):
    crossing_segs_sample = utils.sample_crossings(pos, G, k2i, sampleSize, sampleOn)
#     if len(crossing_segs_sample) < sampleSize*0.1:
#         crossing_segs_sample = utils.sample_crossings(pos, G, k2i, sampleSize, sampleOn='crossings')
        
    if len(crossing_segs_sample) > 0:
        pos_segs = pos[crossing_segs_sample.flatten()].view(-1,4,2)
        w = (torch.rand(pos_segs.shape[0], 2, 1)-0.5).requires_grad_(True)
        b = (torch.rand(pos_segs.shape[0], 1, 1)-0.5).requires_grad_(True)
        relu = nn.ReLU()
        o = optim.SGD([w,b], lr=0.01, momentum=0.5, nesterov=True)
        for _ in range(niter):
            pred = pos_segs.detach() @ w + b
            ## assume labels of nodes in the first edges are -1
            ## now flip the pred of those nodes so that now we want every pred to be +1
            pred[:,:2,:] = -pred[:,:2,:]
            
            loss_svm = relu(1-pred).sum() + reg_coef * w.pow(2).sum()
            o.zero_grad()
            loss_svm.backward()
            o.step()
        pred = pos_segs @ w.detach() + b.detach()
    
        pred[:,:2,:] = -pred[:,:2,:] 
        loss_crossing = relu(1-pred).sum()
        return loss_crossing
    else:
        ##return dummy loss
        return (pos[0,0]*0).sum()
    
    



def angular_resolution(pos, G, k2i, sampleSize=2):
    samples = utils.sample_nodes(G, sampleSize)
    neighbors = [list(G.neighbors(s)) for s in samples]
    sampleIndices = [k2i[s] for s in samples]
    neighborIndices = [[k2i[n] for n in nei] for nei in neighbors]
    
    samples = pos[sampleIndices]
    neighbors = [pos[nei] for nei in neighborIndices]
    
    rays = [nei-sam for nei,sam in zip(neighbors, samples) if len(nei)>1]
    angles = [utils.get_angles(rs) for rs in rays]
    if len(angles) > 0:
        loss = sum([torch.exp(-a*len(a)).sum() for a in angles])
#         loss = sum([(a - np.pi*20/len(a)).pow(2).sum() for a in angles])
    else:
        loss = pos[0,0]*0##dummy output
    return loss




def gabriel(pos, G, k2i, sampleSize):
    edges = utils.sample_edges(G, sampleSize)
    nodes = utils.sample_nodes(G, sampleSize)
    m,n = len(nodes), len(edges)
    
    edges = np.array([(k2i[e0], k2i[e1]) for e0,e1 in edges])
    nodes = np.array([k2i[n] for n in nodes])
    node_pos = pos[nodes]
    edge_pos = pos[edges.flatten()].reshape([-1,2,2])
    centers = edge_pos.mean(1)
    radii = (edge_pos[:,0,:] - edge_pos[:,1,:]).norm(dim=1, keepdim=True)/2
    
    centers = centers.repeat(1,m).view(-1, 2)
    radii = radii.repeat(1,m).view(-1, 1)
    node_pos = node_pos.repeat(n,1)
    
    relu = nn.ReLU()
#     print((node_pos-centers).norm(dim=1))
    loss = relu(radii - (node_pos-centers).norm(dim=1)).pow(2)
    loss = loss.sum()
    return loss


def crossing_angle_maximization(pos, G, k2i, i2k, sampleSize, sampleOn='edges'):
    edge_list = list(G.edges)
    if sampleOn == 'edges':
        sample_indices = np.random.choice(len(edge_list), sampleSize, replace=False)
        edge_samples = [edge_list[i] for i in sample_indices]
        crossing_segs_sample = utils.find_crossings(pos, edge_samples, k2i)
        
    elif sampleOn == 'crossings':
        crossing_segs = utils.find_crossings(pos, edge_list, k2i)
        crossing_count = crossing_segs.shape[0]
        sample_indices = np.random.choice(crossing_count, min(sampleSize,crossing_count), replace=False)
        crossing_segs_sample = crossing_segs[sample_indices]

    if len(crossing_segs_sample) > 0:
        pos_segs = pos[crossing_segs_sample.flatten()].view(-1,4,2) #torch.stack([torch.stack([pos[i],pos[j],pos[k],pos[l]]) for i,j,k,l in crossing_segs_sample])
        v1 = pos_segs[:,1] - pos_segs[:,0]
        v2 = pos_segs[:,3] - pos_segs[:,2]
        cosSim = torch.nn.CosineSimilarity()
        return (cosSim(v1, v2)**2).sum()
    else:
        return (pos[0,0]*0).sum()##dummy loss
    
    
def aspect_ratio(pos, sampleSize, 
                 angles=torch.arange(7,dtype=torch.float)/7*(np.pi/2), 
                 target_width_to_height=[1,1], 
                 scale=0.1):
    
    if sampleSize is not None:
        n = pos.shape[0]
        i = np.random.choice(n, min(n,sampleSize), replace=False)
        samples = pos[i,:]
    else:
        samples = pos
        
    mean = samples.mean(dim=0, keepdim=True)
#     print(mean)
    samples -= mean
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rot = torch.stack([cos, sin, -sin, cos], 1).view(len(angles), 2, 2)
    
    samples = samples.matmul(rot)

    softmax = nn.Softmax(dim=1)
    max_hat = (softmax(samples*scale) * samples).sum(1)
    min_hat = (softmax(-samples*scale) * samples).sum(1)
    
    w = max_hat[:,0] - min_hat[:,0]
    h = max_hat[:,1] - min_hat[:,1]
    estimate = torch.stack([w,h], 1)
    estimate /= estimate.sum(1, keepdim=True)
#     print(estimate)
    target = torch.tensor(target_width_to_height, dtype=torch.float)
    target /= target.sum()
    target = target.repeat(len(angles), 1)
    bce = nn.BCELoss(reduction='mean')
    return bce(estimate, target)



def vertex_resolution(pos, sampleSize=None, target=0.1):
    pairwiseDistance = nn.PairwiseDistance()
    relu = nn.ReLU()
    softmax = nn.Softmax(dim=0)
    softmin = nn.Softmin(dim=0)
    
    n = pos.shape[0]
    if sampleSize is not None:
        i = np.random.choice(n, min(n,sampleSize), replace=False)
        samples = pos[i,:]
    else:
        samples = pos
    m = samples.shape[0]
    a = samples.repeat([1,m]).view(-1,2)
    b = samples.repeat([m,1])
    pdist = pairwiseDistance(a, b)
#     dmax = (softmax(pdist)*pdist).sum().detach()
    dmax = pdist.max().detach()
    targetDist = target*dmax
    
#     loss = len(pdist)*(softmin(pdist).detach() * relu((targetDist - pdist)/targetDist)).sum()
#     loss = relu((targetDist - pdist)/targetDist).sum()
    loss = relu(1 - pdist/targetDist).sum()
    return loss


def neighborhood_preseration(pos, G, adj, k2i, i2k, 
                             n_roots=2, depth_limit=2, 
                             neg_sample_rate=0.5, 
                             device='cpu'):
    
    pos_samples = []
    for _ in range(n_roots):
        root = i2k[random.randint(0, len(G)-1)]
        G_sub = nx.bfs_tree(G, root, depth_limit=depth_limit)
        pos_samples += [k2i[n] for n in G_sub.nodes]
    pos_samples = sorted(set(pos_samples))
    
    n_neg_samples = int(neg_sample_rate * len(pos_samples))
    neg_samples = [random.randint(0, len(G)-1) for _ in range(n_neg_samples)] ##negative samples
    
    samples = sorted(set(
        pos_samples + neg_samples
    )) ##remove duplicates
    
    
    pos = pos[samples,:]
    adj = adj[samples,:][:, samples]
    
    n,m = pos.shape
    x = pos
    
    ## k_dist
    degrees = adj.sum(1).numpy().astype(np.int64)
    max_degree = degrees.max().item()
    n_neighbors = max(2, min(max_degree+1, n))

    n_trees = min(64, 5 + int(round((n) ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(n))))
    
    knn_search_index = NNDescent(
        x.detach().numpy(),
        n_neighbors=n_neighbors,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    
    kmax = knn_dists.shape[1]-1
    k_dist = np.array([
        ( knn_dists[i,min(kmax, k)] + knn_dists[i,min(kmax, k+1)] ) / 2 
        for i,k in enumerate(degrees)
    ])

    ## pdist
    x0 = x.repeat(1, n).view(-1,m)
    x1 = x.repeat(n, 1)
    pdist = nn.PairwiseDistance()(x0, x1).view(n, n)


    ## loss 
    pred = torch.from_numpy(k_dist.astype(np.float32)).view(-1,1) - pdist
    target = adj + torch.eye(adj.shape[0], device=device)
    loss = L.lovasz_hinge(pred, target)
    return loss




def edge_uniformity(pos, G, k2i, sampleSize=None):
    n,m = pos.shape[0], pos.shape[1]

    if sampleSize is not None:
        edges = random.sample(G.edges, sampleSize)
    else:
        edges = G.edges

    sourceIndices, targetIndices = zip(*[ [k2i[e0], k2i[e1]] for e0,e1 in edges])
    source = pos[sourceIndices,:]
    target = pos[targetIndices,:]
    edgeLengths = (source-target).norm(dim=1) 
    eu = edgeLengths.std()
    return eu




def stress(pos, D, W, sampleSize=None, samples=None, reduce='sum'):
    if samples is None:
        n,m = pos.shape[0], pos.shape[1]
        if sampleSize is not None:
            i0 = np.random.choice(n, sampleSize)
            i1 = np.random.choice(n, sampleSize)
            x0 = pos[i0,:]
            x1 = pos[i1,:]
            D = torch.tensor([D[i,j] for i, j in zip(i0, i1)])
            W = torch.tensor([W[i,j] for i, j in zip(i0, i1)])
        else:
            x0 = pos.repeat(1, n).view(-1,m)
            x1 = pos.repeat(n, 1)
            D = D.view(-1)
            W = W.view(-1)
    else:
        x0 = pos[samples[:,0],:]
        x1 = pos[samples[:,1],:]
        D = torch.tensor([D[i,j] for i, j in samples])
        W = torch.tensor([W[i,j] for i, j in samples])
    pdist = nn.PairwiseDistance()(x0, x1)
#     wbound = (1/4 * diff.abs().min()).item()
#     W.clamp_(0, wbound)
    
    res = W*(pdist-D)**2
    
    if reduce == 'sum':
        return res.sum()
    elif reduce == 'mean':
        return res.mean()



