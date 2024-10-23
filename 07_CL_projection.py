# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:11:58 2024

@author: fische_r
"""

import numpy as np
import skimage.io
import cupy as cp
import cupyx.scipy.ndimage as GPUndimage
# import cucim.skimage as GPUskimage
from scipy import ndimage
from cucim.skimage.morphology import ball as GPUball
from time import sleep 
import os
from joblib import Parallel, delayed
import xarray as xr

temppath = '/mnt/SSD/fische_r/tmp'
toppath = '/mnt/nas_nanotomData/CT_Data_PSI/FR54/2023_COELY_postmortem'
outpath = os.path.join(toppath, 'ACL_projections_v2')
if not os.path.exists(outpath):
    os.mkdir(outpath)


bins=np.arange(-0.2,1.8,0.025)

def find_free_GPU_memory(gpu_id, limit=0.75, num_GPU = 5):
    free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
    
    while free<limit:
        gpu_id = (gpu_id+1)%num_GPU #for all available GPUs (5 on mpc2959)
        sleep(2)
        free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
        
    return gpu_id


def pad_and_close3D(im, gpu_id, radius=37):
    shp = np.array(im.shape)+2*radius
    im_padded = np.zeros(shp, dtype=bool)
    im_padded[radius:-radius,radius:-radius,radius:-radius] = im
    
    # im_padded = skimage.morphology.binary_closing(im_padded, ball(radius))
    # use gpu for speed up
    with cp.cuda.Device(gpu_id):
        imgpu = cp.array(im_padded)
        imgpu =  GPUndimage.binary_opening(imgpu) #remove spurious pixels
    
        #select largest objec to avoid PTL pixels (actually necessary?)
            
        imgpu =  GPUndimage.binary_closing(imgpu, structure=GPUball(radius-2))  #-2 to avoid touching image bounds
        # imgpu =  GPUndimage.binary_erosion(imgpu)
        im_padded = cp.asnumpy(imgpu)
    
        del imgpu
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    
    im = im_padded[radius:-radius,radius:-radius,radius:-radius]
    return im

def get_surface(filled_GDL):
    # recycled surface extraction from industry project
    GDL_pos_1 = filled_GDL.argmax(axis=1)
    shp = filled_GDL.shape
    GDL_pos_2 = shp[1]-np.flip(filled_GDL, axis=1).argmax(axis=1)
    med_1 = np.median(GDL_pos_1)
    med_2 = np.median(GDL_pos_2)
    GDL_pos_1[np.abs(GDL_pos_1-med_1)>shp[1]/2] = med_1
    GDL_pos_2[np.abs(GDL_pos_2-med_2)>shp[1]/2] = med_2
    return GDL_pos_1, GDL_pos_2

def CL_trace(im):
    imbin = np.zeros(im.shape, dtype=bool)
    im[~np.isfinite(im)] = 0

    # find highest grayvalue in trace to get a skeleton following the CL
    imargmax = np.argmax(im, axis=1)
    imargmax = ndimage.median_filter(imargmax, size=2)
    for x in range(im.shape[0]):
        for z in range(im.shape[2]):
            y = imargmax[x,z]
            imbin[x,y,z] = True
            
    # dilate
    imbin = ndimage.binary_dilation(imbin)

    #fill gaps
    pos_1, pos_2 = get_surface(imbin)
    for x in range(im.shape[0]):
        for z in range(im.shape[2]):
            y1 = pos_1[x,z]
            y2 = pos_2[x,z]
            imbin[x,y1:y2,z] = True
    
    return imbin

def load_grayvalue(ser, sample, stage):
    proc = '_registered.tif'
    if ser in ['A','B', 'Z']:
        proc = '_rotcrop.tif'
    impath = os.path.join(toppath, ser+'_series',ser+'_'+sample,ser+'_'+sample+'_'+stage+proc)
    im = np.transpose(skimage.io.imread(impath)[100:1100,50:550,:], (1,2,0))  #do the transpose and cropping to match like the weka segmented image in jupyter to not mess up things
    return im

def load_segmented(ser, sample, stage):
    sertop = ser
    if ser in ['A','B']:
        sertop = 'ABCZ'   #A,B cannot be registered with PTL and did not get segmented again like the others
    impath = os.path.join(toppath, sertop+'_CL_segmented', ser+'_'+sample+'_'+stage+'__ACL_segmented.tif')
    im = np.transpose(skimage.io.imread(impath), (1,2,0))<1
    return im

def CL_hist(im, bins = bins):
    return np.histogram(im, bins=bins)[0]
    
def project_CL(im, imseg, gpu_id):
    im_bu = im.copy()
    im_trace = pad_and_close3D(imseg, gpu_id)
    im = im.astype(float)
    im[~im_trace] = np.nan
    
    # v2 more eleaborate trace
    imbin = CL_trace(im)
    im = im_bu.astype(float)
    im[~imbin] = np.nan
   
    
    proj = np.nanmean(im, axis=1)
    hist = CL_hist(im)
    return proj, hist

def extract_samples(series):
    samples = []
    for ser in series:
        print(ser)
        sertop = ser
        if ser in ['A','B']:
            sertop = 'ABCZ'
        serlist = os.listdir(os.path.join(toppath, sertop+'_CL_segmented'))
        for filename in serlist:
            if filename[0] in series:
                sample_name = filename.split('__')[0]
           # if sample_name[:3] == 'G_5': 
                if ser in ['A','B']:
                    if sample_name[0] == 'C': continue
                    if ser == 'A' and sample_name[0] == 'B': continue
                    if ser == 'B' and sample_name[0] == 'A': continue
                print(sample_name)
                samples.append(sample_name)
    return samples

def sample_function(sample_name, i):
    sample_name = sample_name.split('_')
    ser = sample_name[0]
    sample = sample_name[1]
    stage = sample_name[2]
    if len(sample_name)>3:
        stage = stage+'_'+sample_name[3]

    im = load_grayvalue(ser, sample, stage)
    imseg = load_segmented(ser, sample, stage)
    
    gpu_id = find_free_GPU_memory(i%5)
    proj,  hist = project_CL(im, imseg, gpu_id)
    
    path = os.path.join(outpath, ser+'_'+sample+'_'+stage+'_ACL_projection.tif')
    skimage.io.imsave(path, proj)
    return hist


series = ['A','B','C','D','E','F', 'G']
# series = ['G']
samples = extract_samples(series)
print(samples)
results = Parallel(n_jobs = 16, temp_folder=temppath)(delayed(sample_function)(samples[i], i) for i in range(len(samples)))

#create sample list
results = np.stack(results)

np.save(os.path.join(toppath, 'volume_hist_dump_v2_clean.npy'), results)

data = xr.Dataset({'volume_hist': (['sample', 'bin'], results)},
                   coords = {'sample': samples,
				'bin': bins[:-1]}
     )

data.to_netcdf(os.path.join(toppath, 'CL_grayvalue_histograms_v2_clean.nc'))
