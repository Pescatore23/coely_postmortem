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

def crop_cathode_CL(im):
    area = np.argmax(im>0.3, axis=2)
    med = np.median(area).astype(int)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            crop = area[x,y]
            if np.abs(crop-med)>10:
                crop = med
            crop = crop+10
            im[x,y,:crop] = 0

    return im

def unsharp_mask_as_IJ(im, sigma, weight):
    # https://imagej.net/ij/developer/source/ij/plugin/filter/UnsharpMask.java.html
    blur = ndimage.gaussian_filter(im, sigma)
    im = (im - weight*blur)/(1-weight)
    return im

def crop_and_normalize(impath, crop=True):
    im = skimage.io.imread(impath)
    im = im[100:1100,50:550,:] #crop to active area
    if crop: im = crop_cathode_CL(im)
    im = unsharp_mask_as_IJ(im, 1, 0.6)
    
    #normalize image to similar grayvalues for all samples
    immax = im.max(axis=2)
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
        impath = os.path.join(sample_path, imroot+'rotcrop.tif')
        option = False
        
        if series == 'C' and sample == '4':
            option = True
        if series == 'Z' and sample == '4':
            option = True
        
        if option:
            impath = os.path.join(sample_path, imroot+'rotcrop_CCL_manually_removed.tif')
            im = crop_and_normalize(impath, crop=False)
        else:
            im = crop_and_normalize(impath)
        
        outpath = os.path.join(temppath, 'normalized')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        skimage.io.imsave(os.path.join(outpath, imroot+'_normalized.tif'), im)


def series_function(series, n_jobs = 8):
    series_path = os.path.join(toppath, series+'_series')
    folders = os.listdir(series_path)
    samples = []
    for sample in folders:
        samples.append(sample.split('_')[-1])

    Parallel(n_jobs = n_jobs, temp_folder=temppath)(delayed(sample_function)(series, sample) for sample in samples)
        
        
series = ['A', 'B', 'C', 'Z']
# series = ['C', 'Z']
Parallel(n_jobs = 4, temp_folder=temppath)(delayed(series_function)(ser) for ser in series)
        
    
    
    

