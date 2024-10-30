# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:02:21 2024

@author: fische_r
"""

import numpy as np #numpy version 1.26.4
import os 
import skimage.io #scikit-image version 0.22.0
from joblib import Parallel, delayed #joblib version 1.3.2
from scipy import ndimage #scipy version 1.12.0

toppath = '/mpc/homes/fische_r/nanotom_data/2023_COELY_postmortem'
temppath = '/mnt/SSD/fische_r/tmp'

def crop_cathode_CL(im, cropfix):
    area = np.argmax(im>0.3, axis=2)
    med = np.median(area).astype(int)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            crop = area[x,y]
            if np.abs(crop-med)>cropfix: #30 for D and E series, was 10 before
                crop = med
            crop = crop+10
            im[x,y,:crop] = 0

    return im

def unsharp_mask_as_IJ(im, sigma, weight):
    # https://imagej.net/ij/developer/source/ij/plugin/filter/UnsharpMask.java.html
    blur = ndimage.gaussian_filter(im, sigma)
    im = (im - weight*blur)/(1-weight)
    return im

def crop_and_normalize(impath,CLpath, ser, crop=True):
    im = skimage.io.imread(impath)
    im = im[100:1100,50:550,:] #crop to active area
    cropfix = 30
    if ser == 'C': cropfix = 10
    if crop: im = crop_cathode_CL(im, cropfix)
    im = unsharp_mask_as_IJ(im, 1, 0.6)
    
    #normalize image to similar grayvalues for all samples
    immax = im.max(axis=2)
    skimage.io.imsave(CLpath, immax)
    med = np.median(immax)
    diff = med - 0.8
    im = im - diff
    
    return im
    
    
def sample_function(series, sample):
    sample_path = os.path.join(toppath, series+'_series', series+'_'+sample)
    
    files = os.listdir(sample_path)
    stages = []
    for file in files:
        splitfile = file.split('_')
        if splitfile[-1] == 'rotcrop.tif':
            stage = ''.join([i+'_' for i in splitfile[2:-1]])
            stages.append(stage)
            
    for stage in stages:
        imroot = series+'_'+sample+'_'+stage
        # impath = os.path.join(sample_path, imroot+'rotcrop.tif')
        impath = os.path.join(sample_path, imroot+'registered.tif')
        outpathCL = os.path.join(sample_path, 'CL_extraction')
        if not os.path.exists(outpathCL):
            os.mkdir(outpathCL)
        CLpath = os.path.join(outpathCL, imroot+'_extracted_CL.tif')
        option = False
        
        
        if series == 'Z' and sample == '4':
            option = True
        
        if option:
            impath = os.path.join(sample_path, imroot+'rotcrop_CCL_manually_removed.tif')
            im = crop_and_normalize(impath, CLpath, series, crop=False)
        else:
            im = crop_and_normalize(impath, CLpath, series)
        
        outpath = os.path.join(toppath, series+'_normalized')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        skimage.io.imsave(os.path.join(outpath, imroot+'_normalized.tif'), im)


def series_function(series, n_jobs = 8):
    series_path = os.path.join(toppath, series+'_series')
    folders = os.listdir(series_path)
    samples = []
    for sample in folders:
        # if sample[0] == 'D':
        #     if not sample == 'D_5': continue
        # if sample[0] == 'F':
        #     if sample == 'F_1': continue
        # if sample in ['E_1','E_2', 'F_2']: continue
        # # if sample[0] == 'G':
        if not sample == 'Z_3': continue
        samples.append(sample.split('_')[-1])

    Parallel(n_jobs = n_jobs, temp_folder=temppath)(delayed(sample_function)(series, sample) for sample in samples)
        
        
series = ['A', 'B', 'C', 'Z']
# series = ['C', 'Z']
series = ['D', 'E']
series = ['C', 'D', 'E', 'F', 'G']
series = ['Z']
Parallel(n_jobs = 8, temp_folder=temppath)(delayed(series_function)(ser) for ser in series)
        
    
    
    

