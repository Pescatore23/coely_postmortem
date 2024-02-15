# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:34:55 2024

adapted from coely_tomcat 06b_analysis_CL_crumbling (https://gitlab.psi.ch/fcsd_5422/coely_tomcat_processing)

@author: fische_r
"""

import os
from joblib import Parallel, delayed
import numpy as np
import cupy as cp
from time import sleep
from cucim.skimage.morphology import ball as GPUball
import cupyx.scipy.ndimage as GPUndimage
import skimage.io



temppath = '/mnt/SSD/fische_r/tmp'
toppath = '/mnt/nas_nanotomData/CT_Data_PSI/FR54/2023_COELY_postmortem'
segpath = os.path.join(toppath, 'ABCZ_CL_segmented')
outpath = os.path.join(toppath, 'ABCZ_CLhole_segmented')
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
    scan_pos = range(0, mask.shape[1]-1, 2)
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
    #fileroot = ''.join([i+'_' for i in splitfile[:-1]])
#    fileroot = fileroot[:-1]
    fileroot = splitfile[0]
    
    im = skimage.io.imread(path)
    im = im.transpose(0,2,1)[:,40:,:]
    im = im>0
    
    diff_mean = hole_scan_3D(im)
    holes1 = diff_mean[:,0,:]>-0.5
    holes2 = diff_mean[:,-1,:]>-0.5
    # CLhole_area = holes1*1+holes2*2
    CLholes = holes1.sum()/2+holes2.sum()/2
    
    outfile = os.path.join(outpath, fileroot+'__ACL_holes.tif')
    skimage.io.imsave(outfile, diff_mean)
    return CLholes


files = os.listdir(segpath)

Clholes = image_function(files[0])
print(Clholes)
    
    
    
    
