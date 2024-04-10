import os, sys 
import numpy as np
import nibabel as nib 
import pingouin as pg 
import pycircstat as circ
import hcp_utils as hcp 
import networkx as nx
from neuromaps import datasets

fslr = datasets.fetch_atlas(atlas='fslr', density='32k')
slh, srh = fslr['midthickness']
lh_vert, lh_face = nib.load(slh).agg_data()
rh_vert, rh_face = nib.load(srh).agg_data()

def v_cluster_labeling(data, mask, thres, neg = False):

    if mask != None:
        data_lr = np.zeros(32492*2)
        data_lr[mask] = data
    else:
        data_lr = data.copy()

    if neg:
        data_lr[data_lr > thres] = 0
    else:
        data_lr[data_lr < thres] = 0
    
    lr_mask = np.array(data_lr!=0, dtype=int)
    lh_mask = lr_mask[:32492]
    rh_mask = lr_mask[32492:]

    # left hemis 
    lh_mask_indices = np.where(lh_mask==1)[0]
    lh_mask_vertice = lh_vert[lh_mask_indices,:]

    G = nx.Graph()
    for i, vertex in enumerate(lh_vert):
        if i in lh_mask_indices:
            G.add_node(i, coords=vertex)
    for face in lh_face:
        if all(v in lh_mask_indices for v in face):
            G.add_edge(face[0], face[1])
            G.add_edge(face[1], face[2])
            G.add_edge(face[2], face[0])

    ccsL  = nx.connected_components(G) 
    lh_clus = []
    lh_clusize = []
    for ccl in ccsL:
        lh_clus.append(list(ccl))
        lh_clusize.append(len(ccl))
    lh_clusize = np.asarray(lh_clusize)

    # right hemis 
    rh_mask_indices = np.where(rh_mask==1)[0]
    rh_mask_vertice = rh_vert[rh_mask_indices,:]

    G = nx.Graph()
    for i, vertex in enumerate(rh_vert):
        if i in rh_mask_indices:
            G.add_node(i, coords=vertex)
    for face in rh_face:
        if all(v in rh_mask_indices for v in face):
            G.add_edge(face[0], face[1])
            G.add_edge(face[1], face[2])
            G.add_edge(face[2], face[0])

    ccsR  = nx.connected_components(G) 
    rh_clus = []
    rh_clusize = []
    for ccr in ccsR:
        rh_clus.append(list(ccr))
        rh_clusize.append(len(ccr))
    rh_clusize = np.asarray(rh_clusize)   

    return lh_clus, lh_clusize, rh_clus, rh_clusize


def v_vector_comp(mat1, mat2):
    #   calculating the variance of two angles and assign a sign to the variance
    ##  mat1 (and mat2): subjects x vertices
    ### input is the angle transformed from sensory betas
    
    theta_dif = circ.cdiff(mat1, mat2)
    theta_dif_sign = np.zeros((np.shape(theta_dif)))
    theta_dif_sign[theta_dif>0] = 1   
    theta_dif_sign[theta_dif<0] = -1

    theta_com = []
    for s in range(mat1.shape[0]):
        theta_pack = np.vstack((mat1[s,:], mat2[s,:]))
        theta_diff = 1 - pg.circ_r(theta_pack, axis=0)
        theta_com.append(theta_diff)
    theta_com = np.asarray(theta_com)
    theta_com = theta_com * theta_dif_sign

    return theta_com

def v_perm_clusize(theta_com, vertP_thres, n_perm):

    n_sub = dif.shape[0]
    n_ver = dif.shape[1]

    maxClus = np.zeros(n_perm)
    for n in range(n_perm):
        rands = np.random.randint(2, size=n_sub)
        rand_dif = theta_com.copy()[rands==1,:] * -1
        lh_clus, lh_clusize, rh_clus, rh_clusize = v_cluster_labeling(rand_dif, None, vertP_thres, neg = False) 
        maxClus[n] = np.max(np.concatenate((lh_clusize, rh_clusize)))

    return maxClus