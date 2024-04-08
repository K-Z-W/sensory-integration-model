import os 
import numpy as np
import pingouin as pg 
import nibabel as nib
import scipy.stats as ss
import hcp_utils as hcp 
from sklearn.linear_model import LinearRegression as LReg

parc_l = hcp.mmp.map_all[hcp.struct.cortex_left]
parc_r = hcp.mmp.map_all[hcp.struct.cortex_right]
parc   = hcp.mmp.map_all[hcp.struct.cortex]


def v_ts_nnls_vertex(ts): 
    # non-negative linear regression
    # with time-seris of V1, S1, and A1 as predictors
    
    ts_v1 = ts[(parc==1)|(parc==181),:].mean(axis=0)
    ts_s1 = ts[(parc==9)|(parc==51)|(parc==52)|(parc==53)|(parc==189)|(parc==231)|(parc==232)|(parc==233)].mean(axis=0)
    ts_a1 = ts[(parc==24)|(parc==204),:].mean(axis=0)

    ts_x = np.vstack((ts_v1, ts_s1, ts_a1)).T
    ts_y = ts.T

    reg_nnls = LReg(positive=True)
    res_nnls = reg_nnls.fit(ts_x, ts_y)

    rgba = np.zeros((4,ts.shape[0]))
    rgba[:3,:] = res_nnls.coef_.T
    y_bar = ts_y.mean(axis=0)
    y_hat = res_nnls.predict(ts_x)
    ss_total = np.sum((ts_y - y_bar)**2, axis=0)
    ss_exp   = np.sum((y_hat - y_bar)**2, axis=0)
    rgba[3, :] = ss_exp / ss_total

    return rgba


def v_rgba2hue(rgba):
    # rgba here is an [m x n] matrix, the first 3 rows (from m) should be betas we are going to use
    rgbv = rgba[:3,:]
    H    = np.zeros(rgbv.shape[1])

    delta = np.max(rgbv, axis=0) - np.min(rgbv, axis=0)
    del_0 = delta==0
    H[del_0] = 0

    rgbv_pos = rgbv[:3, ~del_0]
    del_pos = delta[~del_0]
    ind_max = np.argmax(rgbv_pos, axis=0)
    Hp = np.zeros(rgbv_pos.shape[1])
    Hp[ind_max==0] = 60 * (((rgbv_pos[1,ind_max==0] - rgbv_pos[2,ind_max==0])/del_pos[ind_max==0]))
    Hp[ind_max==1] = 60 * (((rgbv_pos[2,ind_max==1] - rgbv_pos[0,ind_max==1])/del_pos[ind_max==1])+2)
    Hp[ind_max==2] = 60 * (((rgbv_pos[0,ind_max==2] - rgbv_pos[1,ind_max==2])/del_pos[ind_max==2])+4)

    H[~del_0] = Hp

    return H

def v_hsv_model_rgba(rgba):
    # rgba here is an [m x n] matrix, the first 3 rows (from m) should be betas from primary sensory cortex, and the 4th row is the variance explained by three sensory predictors in the linear regression.
    hue = v_rgba2hue(rgba)

    rd = rgba[3,:]
    rd = ss.rankdata(rd)
    rd = (rd - rd.min()) / (rd.max() - rd.min())

    hsv  = np.zeros((rgba.shape[1], 3))
    hsv[:,0] = hue/360
    hsv[:,1] = rd
    hsv[:,2] = 0.86
    rgb      = color.hsv_to_rgb(hsv)
    theta    = 2 * np.pi * hue/360

    return hsv, rgb, theta, rd

def v_hsv_model_rgba_indiv(rgba):
    # rgba here is an [s x m x n] matrix, s for subjects, the first 3 rows (from m) should be betas from primary sensory cortex, and the 4th row (from m) is the variance explained by three sensory predictors in the linear regression.
    n_sub, _, n_ver = rgba.shape 
    th_indiv = []
    rd_indiv = []

    for s in tqdm(range(n_sub)):
        rgbv = rgba[s,:,:]
        _,_,th,rd = v_hsv_model_rgba(rgbv)
        th_indiv.append(th)
        rd_indiv.append(rd)

    th_indiv = np.asarray(th_indiv)
    rd_indiv = np.asarray(rd_indiv)

    ex_group = rgba[:,3,:].mean(axis=0)
    ex_group = ss.rankdata(ex_group)
    rd_group = (ex_group - ex_group.min())/(ex_group.max() - ex_group.min())

    th_group = pg.circ_mean(th_indiv, axis=0)
    th_group[th_group < 0] = th_group[th_group<0] + 2*np.pi
    hsv = np.zeros((n_ver, 3))
    hue = th_group/(2*np.pi)
    hsv[:,0] = hue
    hsv[:,1] = rd_group
    hsv[:,2] = 0.86
    rgb_group = color.hsv_to_rgb(hsv)

    return th_indiv, rd_indiv, th_group, rd_group, rgb_group

# Following is an example:
# rgba_mv is a 167 x 4 x 59412 matrix,
# which means 167 subjects, 3 kinds of betas + variance explained, and 59412 vertices.
# th_grp_mv is the group level angles transformed from betas;
# th_ind_mv is the individual level angles transformed from betas;
# rd is for magnitude, or rescaled ranking variance explained;
# color is for visualization, with group angles as hue, magnitude as saturation, and 0.86 as lightness.
#-------------------------------------------------------------------------------------------
# rgba_mv = np.load(dpath + '/rgba_7t_mv.npy')
# rgba_mv.shape()
# --> (167, 4, 59412)
# th_ind_mv, rd_ind_mv, th_grp_mv, rd_grp_mv, color_grp_mv = v_hsv_model_rgba_indiv(rgba_mv) 
#-------------------------------------------------------------------------------------------
