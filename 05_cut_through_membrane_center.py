# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:10:49 2024

this script cuts through the membrane to yield a layer at the middle between the two catalyst layers
regardless of orientation or membrane distortion like bending

does not work for PTL samples ---> needs different solution

@author: fische_r
"""

import os
import skimage.io
import scipy as sp
import scipy.signal
import scipy.ndimage
import numpy as np
from joblib import Parallel, delayed

toppath = '/mpc/homes/fische_r/nanotom_data/2023_COELY_postmortem'
temppath = '/mnt/SSD/fische_r/tmp'

def search_crude_CL(im):
    med = np.median(im)
    shp = im.shape
    CL = np.zeros((shp[0],shp[1],2), dtype = int)
    for x in range(shp[0]):
        for y in range(shp[1]):
            peaks, props = sp.signal.find_peaks(im[x,y,:], height=6.25*med, distance=12, prominence=4.5*med)
            LP = len(peaks)
            if LP>0 and LP<3:
                CL[x,y,-LP:] = peaks
    return CL 


def find_center_surface(CL):
    CL0 = sp.ndimage.median_filter(CL[:,:,0], size = 10)
    CL1 = sp.ndimage.median_filter(CL[:,:,1], size = 10)
    IFcoords = np.uint16((CL0+CL1)/2)
    return IFcoords


def extract_center_face(im, IFcoords):
    shp = IFcoords.shape
    interface = np.zeros(shp)
    for x in range(shp[0]):
        for y in range(shp[1]):
            z = IFcoords[x,y]
            interface[x,y] = im[x,y,z]
    return interface


def cut_through_membrane_center(im):
    CL = search_crude_CL(im)
    IFcoords = find_center_surface(CL)
    interface = extract_center_face(im, IFcoords)
    return interface


def sample_function(series, sample):
    sample_path = os.path.join(toppath, series+'_series', series+'_'+sample)
    outpath = os.path.join(toppath, 'membrane_cut')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
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
        im = skimage.io.imread(impath)
        im = im[100:1100,50:550,:]
        im = cut_through_membrane_center(im)
        skimage.io.imsave(os.path.join(outpath, imroot+'_membrane_cut.tif'), im)
        
        
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
        
        