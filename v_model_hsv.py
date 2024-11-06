import os 
import numpy as np
import nibabel as nib 
import hcp_utils as hcp 
import pandas as pd
import matplotlib.colors as color
import scipy.stats as ss
import pycircstat as circ


def v_rgba2hue(rgba):

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
    n_sub, _, n_ver = rgba.shape 
    th_indiv = []
    rd_indiv = []

    for s in range(n_sub):
        rgbv = rgba[s,:,:]
        _,_,th,rd = v_hsv_model_rgba(rgbv)
        th_indiv.append(th)
        rd_indiv.append(rd)

    th_indiv = np.asarray(th_indiv)
    rd_indiv = np.asarray(rd_indiv)
    # rd_group = rd_indiv.mean(axis=0)

    ex_group = rgba[:,3,:].mean(axis=0)
    ex_group = ss.rankdata(ex_group)
    rd_group = (ex_group - ex_group.min())/(ex_group.max() - ex_group.min())

    th_group = circ.descriptive.mean(th_indiv, axis=0)
    th_group[th_group < 0] = th_group[th_group<0] + 2*np.pi
    hsv = np.zeros((n_ver, 3))
    hue = th_group/(2*np.pi)
    hsv[:,0] = hue
    hsv[:,1] = rd_group
    hsv[:,2] = 0.86
    rgb_group = color.hsv_to_rgb(hsv)

    return th_indiv, rd_indiv, th_group, rd_group, rgb_group

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
        theta_diff = 1 - circ.descriptive.resultant_vector_length(theta_pack, axis=0)
        theta_com.append(theta_diff)
    theta_com = np.asarray(theta_com)
    theta_com = theta_com * theta_dif_sign

    return theta_com

def v_save_gii(data, savepath, savename, half=False):

    if half:
        data_lh = nib.gifti.gifti.GiftiImage()
        data_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data, datatype='NIFTI_TYPE_FLOAT32'))

        savename_lh = 'lh.'+savename+'.func.gii'
        nib.save(data_lh, os.path.join(savepath, savename_lh))
    else:
        data_lh = nib.gifti.gifti.GiftiImage()
        data_lh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data[:32492], datatype='NIFTI_TYPE_FLOAT32'))
        data_rh = nib.gifti.gifti.GiftiImage()
        data_rh.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data[32492:], datatype='NIFTI_TYPE_FLOAT32'))

        savename_lh = 'lh.'+savename+'.func.gii'
        nib.save(data_lh, os.path.join(savepath, savename_lh))
        savename_rh = 'rh.'+savename+'.func.gii'
        nib.save(data_rh, os.path.join(savepath, savename_rh))
