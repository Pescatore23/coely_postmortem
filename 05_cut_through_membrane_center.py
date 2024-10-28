# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:10:49 2024

this script cuts through the membrane to yield a layer at the middle between the two catalyst layers
regardless of orientation or membrane distortion like bending

@author: fische_r
"""

import os
import skimage.io #scikit-image version 0.22.0
import scipy as sp #scipy version 1.12.0
import scipy.signal #scipy version 1.12.0
import scipy.ndimage #scipy version 1.12.0
import numpy as np #numpy version 1.26.4
from joblib import Parallel, delayed #joblib version 1.3.2

toppath = '/mpc/homes/fische_r/nanotom_data/2023_COELY_postmortem'
temppath = '/mnt/SSD/fische_r/tmp'

def search_crude_CL(im):
    med = np.median(im)
    shp = im.shape
    CL = np.zeros((shp[0],shp[1],2), dtype = int)
    grad = -np.gradient(np.gradient(im, axis=2), axis=2)
    # grad = np.gradient(im, axis=2)
    for x in range(shp[0]):
        for y in range(shp[1]):
            prof = grad[x,y,:]
            peaks, props = sp.signal.find_peaks(prof, height=6*med/8, distance=15, prominence=med/4) #gives peaks sorted by height (= ACL first)
            LP = len(peaks)
            if LP>0:
                peaks = peaks[:2]
                CL[x,y,-LP:] = peaks #inverse order to put CCL first
    return CL    #, excess_peaks


def find_center_surface(CL, ser, sample):
    # some hard coded limits and median filtering
    # TODO: consider that position for custom BPM is closer to cathode
    
    CL0 = CL[:,:,0] #CCL ?!
    CL1 = CL[:,:,1] #ACL ?!

    #global median filter
    medCL0 = np.median(CL0[CL0>0])
    CL0[CL0<5] = medCL0

    medCL1 = np.median(CL1)
    if ser in 'ABCZ':
        CL1[CL1<medCL0+10] = medCL1
    
    # print(medCL0, medCL1)

    # #local median filter
    CL0 = sp.ndimage.median_filter(CL0, size = 4)
    CL1 = sp.ndimage.median_filter(CL1, size = 4)
    
    diff = (CL1-CL0)/2
    meddiff = np.median(diff)
    diff[diff<10] = meddiff
    diff[diff>60]  = meddiff
    
    # if ser == 'D' or ser == 'E':
    #     diff[diff<10] = meddiff
    #     diff[diff>95]  = meddiff
    #     IFcoords = np.uint16(CL0+30) #20 roughly the thickness of FAA in px, 30 checks if it can find the Nafiuon membrane
    # else:
    #     IFcoords = np.uint16(CL1-diff)
    
 #   if ser not in 'ABCZ':
#        IFcoords = np.uint16(CL1-62) #58 is a little more than the ANfion thickness

    if ser not in 'ABCZ':
        IFcoords = np.uint16(CL1-60)
        mask = CL0[(CL0-IFcoords)<-30]
        IFcoords[mask] = np.uint16(CL0[mask]+30)
        coordmed = np.median(IFcoords)
        IFcoords[IFcoords>200] = coordmed
        IFcoords[IFcoords<10] = coordmed
        IFcoords = sp.ndimage.median_filter(IFcoords, size = 4)

        # print(ser, sample, medcoord)
        
    else:
        IFcoords = np.uint16((CL1+CL0)/2)
        coordmed = np.median(IFcoords)
        IFcoords[IFcoords>80] = coordmed
        IFcoords[IFcoords<30] = coordmed
        IFcoords = sp.ndimage.median_filter(IFcoords, size = 4)
    
    return IFcoords

def extract_center_face(im, IFcoords):

    shp = IFcoords.shape
    interface = np.zeros(shp)
    for x in range(shp[0]):
        for y in range(shp[1]):
            z = IFcoords[x,y]
            interface[x,y] = im[x,y,z-4:z+4].min()
            # interface[x,y] = im[x-fs:x+fs,y-fs:y+fs,z-fs:z+fs].mean()  #small mean filtering

    return interface

def cut_through_membrane_center(im, ser, sample):
    CL = search_crude_CL(im)
    IFcoords = find_center_surface(CL, ser, sample)
    interface = extract_center_face(im, IFcoords)
    return interface, IFcoords

# def cut_through_membrane_center(im):
#     CL = search_crude_CL(im)
#     IFcoords = find_center_surface(CL)
#     interface = extract_center_face(im, IFcoords)
#     return interface


def sample_function(series, sample):
    try:
        sample_path = os.path.join(toppath, series+'_series', series+'_'+sample)
        outpath = os.path.join(toppath, 'membrane_cut')
        outpath2 = os.path.join(toppath, 'membrane_cut_positions')
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        if not os.path.exists(outpath2):
            os.mkdir(outpath2)
        
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
            im, impos = cut_through_membrane_center(im, series, sample)
            skimage.io.imsave(os.path.join(outpath, imroot+'_membrane_cut.tif'), im)
            skimage.io.imsave(os.path.join(outpath2, imroot+'_membrane_cut_position.tif'), impos)
    except:
        print(series+sample+'_failed')
        
        
def series_function(series, n_jobs = 8):
    series_path = os.path.join(toppath, series+'_series')
    folders = os.listdir(series_path)
    samples = []
    for sample in folders:
        samples.append(sample.split('_')[-1])

    Parallel(n_jobs = n_jobs, temp_folder=temppath)(delayed(sample_function)(series, sample) for sample in samples)
    
    
    
    

series = ['A', 'B', 'C','D', 'E', 'F', 'G', 'Z']
#series = ['G']
Parallel(n_jobs = 16, temp_folder=temppath)(delayed(series_function)(ser) for ser in series)
        
        

