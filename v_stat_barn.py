import os, sys 
import numpy as np
import nibabel as nib 
import hcp_utils as hcp 
import networkx as nx

dir_surf = r'/Users/weiwei/BaiduCloud/Work/Paris/HSV/Surfs'
slh = os.path.join(dir_surf, 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii')
srh = os.path.join(dir_surf, 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii')

lh_vert, lh_face = nib.load(slh).agg_data()
rh_vert, rh_face = nib.load(srh).agg_data()

def v_cluster_labeling(data, thres, Neg = False, Abs = False, Half = False):

    if Half:
        data_lr = np.zeros(32492*2)
        data_lr[:32492] = data.copy()
    else:
        data_lr = data.copy()

    if Abs:
        data_lr = abs(data_lr)

    if Neg:
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

def v_clus_max(lh_clusize, rh_clusize):

    if (len(lh_clusize) > 0) & (len(rh_clusize) > 0):
        v_max = np.max((lh_clusize.max(), rh_clusize.max()))
    elif (len(lh_clusize) > 0) & (len(rh_clusize) == 0): 
        v_max = lh_clusize.max()
    elif (len(lh_clusize) == 0) & (len(rh_clusize) > 0): 
        v_max = rh_clusize.max()
    else:
        v_max = 0

    return v_max