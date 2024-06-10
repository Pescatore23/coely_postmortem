# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:34:55 2024

adapted from coely_tomcat 06b_analysis_CL_crumbling (https://gitlab.psi.ch/fcsd_5422/coely_tomcat_processing)

@author: fische_r
"""

import os
from joblib import Parallel, delayed #joblib version 1.3.2
import numpy as np #numpy version 1.26.4
import cupy as cp #cupy version 13.0.0
from time import sleep 
from cucim.skimage.morphology import ball as GPUball #CuCim version 23.10.0
import cupyx.scipy.ndimage as GPUndimage #cupy version 13.0.0
import skimage.io #scikit-image version 0.22.0



temppath = '/mnt/SSD/fische_r/tmp'
toppath = '/mnt/nas_nanotomData/CT_Data_PSI/FR54/2023_COELY_postmortem'
# segpath = os.path.join(toppath, 'ABCZ_CL_segmented')
# outpath = os.path.join(toppath, 'ABCZ_CLhole_segmented')

segpath = os.path.join(toppath, 'DE_CL_segmented')
outpath = os.path.join(toppath, 'DE_CLhole_segmented')
if not os.path.exists(outpath):
    os.makedirs(outpath)

num_GPU = 5

def find_free_GPU_memory(gpu_id, limit=0.5):
    free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
    
    while free<limit:
        gpu_id = (gpu_id+1)%num_GPU #for all available GPUs
        sleep(2)
        free = cp.cuda.Device(gpu_id).mem_info[0]/cp.cuda.Device(gpu_id).mem_info[1]
        
    return gpu_id


def hole_scan_3D(im3D, radius=2, iterations = 100, njobs=5):
    mask = im3D<1
    scan_pos = range(0, mask.shape[1]-1, 3)
    results = Parallel(n_jobs=njobs, temp_folder=temppath)(delayed(diffusion_3D)(mask, scan_pos[i], radius, iterations, i%5) for i in range(len(scan_pos)))
    diffs_stack = np.stack(results, axis=0)
    diff_mean = diffs_stack.mean(axis=0)
        
    return diff_mean
        

def diffusion_3D(mask, free_pos, radius, iterations = 100, gpu_id=0):
    
    # wait for free GPU RAM
    gpu_id = find_free_GPU_memory(gpu_id)
    
    with cp.cuda.Device(gpu_id):
        structure = GPUball(radius)
        mask = cp.array(mask)
        #mask = GPUndimage.binary_erosion(mask, structure=GPUball(1))
        init = cp.zeros(mask.shape, dtype=bool)
        init[:,free_pos,:] = True

        result = init.copy()*1
        reference = init.copy()*1
        flow = init.copy()
        flow2 = init.copy()

        for i in range(iterations):
            flow = GPUndimage.binary_dilation(flow, structure, mask=mask)
            flow2 = GPUndimage.binary_dilation(flow2, structure)
            result = result+flow*1
            reference = reference+flow2*1

        difference = result-reference
        difference = cp.asnumpy(difference)
    
        del result, reference, flow, flow2, init
    
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    return difference


def image_function(file):
    path = os.path.join(segpath, file)
    splitfile = file.split('__')
    fileroot = splitfile[0]
    
    imraw = skimage.io.imread(path)
    
    # if segpath == os.path.join(toppath, 'C_CL_segmented'):
    #     im = im[:,:,40:170] #crop segmented images for G series, because initial crop was wider to encompass the PTL for registartion for the first time
    center = np.argmin(imraw.sum(axis=(0,1)))
    
    im = np.zeros((1000,500,100))
    imcrop = imraw[:,:,center-50:center+50]==0
    shp = imcrop.shape
    z0 = int(shp[2]/2)
    try:
        im[...,50-z0:50+z0] = imcrop 
    except:
        im[...,50-z0+1:50+z0] = imcrop 
    im = im.transpose(0,2,1)#[:,41:,:]
    
    projholes = (~im.max(axis=1)).sum()
    
    
    diff_mean = hole_scan_3D(im)
    holes1 = diff_mean[:,1,:]>-0.3
    holes2 = diff_mean[:,-2,:]>-0.3
    # CLhole_area = holes1*1+holes2*2
    CLholes = holes1.sum()/2+holes2.sum()/2
    
    # outfile = os.path.join(outpath, fileroot+'__ACL_holes.tif')
    # skimage.io.imsave(outfile, diff_mean)
    return fileroot, CLholes, projholes


files = os.listdir(segpath)

datalines = []
datalines.append('sample , CL holes , Cl holes projection \n' )
for file in files:
    fileroot, Clholes, projholes = image_function(file)
    datalines.append(fileroot + ' , '+ str(Clholes)+' , '+str(projholes)+'\n')
    
outfile = os.path.join(outpath, 'CL_hole_area.csv')

with open(outfile, 'w') as f:
    f.writelines(datalines)
    
    
    
    
