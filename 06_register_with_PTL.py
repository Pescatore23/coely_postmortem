# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:47:48 2024

@author: fische_r
"""

import os
import json
import scipy as sp
import scipy.ndimage
import SimpleITK as sitk
import numpy as np
import skimage


toppath = '/mpc/homes/fische_r/nanotom_data/2023_COELY_postmortem'
processing_dict_path = os.path.join(toppath, 'processing_data.json')
processing_dict = json.load(open(processing_dict_path,'r'))

#grayscale range for conversion to 16bit
vmin = -0.12
vmax = 0.7

##### Parameters for the registration   ################
parameterMap = sitk.GetDefaultParameterMap("rigid") #get default settins for rigid registration

# modify some parameters
parameterMap["FixedImagePyramid"] = ["FixedShrinkingImagePyramid"]
parameterMap["MovingImagePyramid"] = ["MovingShrinkingImagePyramid"]
parameterMap["NumberOfResolutions"] = ["2"]
parameterMap["MaximumNumberOfIterations"] = ["5000"]
parameterMap["AutomaticTransformInitialization"] = ["true"]
parameterMap["AutomaticScalesEstimation"] = ["true"]
parameterMap["NewSamplesEveryIteration"] = ["true"]
parameterMap["NewSamplesEveryIteration"] = ["true"]
parameterMap["Interpolator"] = ["BSplineInterpolator"]
parameterMap["NumberOfSamplesForExactGradient"] = ["10000"]
parameterMap["NumberOfSpatialSamples"] = ["10000"]


def load_nanotom(path, pcrSizeX = 0, pcrSizeY = 0, pcrSizeZ = 0):
    if pcrSizeX == 0 or pcrSizeY == 0 or pcrSizeZ == 0:
        pcrpath = path.split('.')[0]+'.pcr'
        with open(pcrpath) as file:
            lines = file.readlines()
        for line in lines:
            splitline = line.split('=')
            if splitline[0] == 'ROI_SizeX':
                pcrSizeX = int(splitline[1])
            if splitline[0] == 'ROI_SizeY':
                pcrSizeY = int(splitline[1])
            if splitline[0] == 'ROI_SizeZ':
                pcrSizeZ = int(splitline[1])
    with open(path,'r') as file:
        file.seek(0)
        im = np.fromfile(file, dtype='<f4').reshape(pcrSizeX,pcrSizeY,pcrSizeZ, order ='F')  
    return im    

def rotate_im(im, angle1, angle2, angle3):
    if np.abs(angle1)>0.25:
        im = sp.ndimage.rotate(im, angle1)
    if np.abs(angle2)>0.25:
        im = sp.ndimage.rotate(im, angle2, axes=(2,0))
    if np.abs(angle3)>0.25:
        im = sp.ndimage.rotate(im, angle3, axes=(2,1))
    return im

def float_to_uint16(im, vmin=vmin, vmax=vmax):
    im = im-vmin
    im[im<0] = 0
    im = im/(vmax-vmin)
    im[im>1] = 1
    im = im*(2**16-1)*0.95
    im = np.uint16(im)
    return im

def get_stage_im(path, series, sample, stage, proc_dict=processing_dict, vmin=vmin, vmax=vmax):
    im = load_nanotom(path)
    angle1, angle2, angle3 = proc_dict[series+'_'+sample]['rotangles'][stage]
    im = rotate_im(im, angle1, angle2, angle3)
    crops = processing_dict[series+'_'+sample]['cropping']
    a,b,c,d,e,f = crops[stage]
    b = a + 300 #modify for larger ROI
    a = a - 10  # -10 minimum: will lead to a=0 for E_1
    im = im[a:b,c:d,e:f]
    #convert to uint16 with the given range
    im = float_to_uint16(im)
    return im



def register_images_general(im_fixed, im_tomove, parameter_map=parameterMap, im_mask=None):

    r"""
    General registration for 2D or 3D images
    This function allows control over the parameter map
         
    Parameters
    ----------     
    im_fixed : numpy.ndarray
        Fixed image
    im_tomove: numpy.ndarray
        Image to register (to im_fixed)
    parameter_map: sitk.ParameterMap
        Parameters map object as obtained from sitk.GetParameterMap
    verbose : bool, optional (default=False)
        Enable/Disable log to console 
    im_mask : numpy.ndarray (default=None)
        Mask for sampling points (required). 
        im_mask has same dimensions as im_fixed 
        The mask ndarray has 1 and 0. 1 values denote the area where
        points for the registration are sampled.    
    Returns
    -------
    im_aligned : numpy.ndarray
        Aligned image having same dimensions of im_tomove
                 
    Notes
    -----
    To have further info on parameter maps, go to:
    https://simpleelastix.readthedocs.io/ParameterMaps.html
    For more details on the different parameters, download:
    https://elastix.lumc.nl/download/elastix_manual_v4.8.pdf
    """
    
    itk_image_fixed = sitk.GetImageFromArray(im_fixed)
    itk_image_tomove = sitk.GetImageFromArray(im_tomove)
           
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(itk_image_fixed)
    elastixImageFilter.SetMovingImage(itk_image_tomove)
    elastixImageFilter.SetParameterMap(parameter_map)
    
    if im_mask is not None:
        itk_image_mask = sitk.GetImageFromArray(im_mask)
        # Casting type just in case
        itk_image_mask = sitk.Cast(itk_image_mask, sitk.sitkUInt8)
        elastixImageFilter.SetFixedMask(itk_image_mask)
    
    elastixImageFilter.SetLogToConsole(False) #disable loggingdd
    elastixImageFilter.Execute()
    
    im_aligned = elastixImageFilter.GetResultImage()
    # Transfor again to numpy
    im_aligned = sitk.GetArrayFromImage(im_aligned)
    return np.uint16(im_aligned) #careful with this line with float images from nanotom
    

def sample_function(series, sample, toppath=toppath):
    # sample paths do work for series D and E, there will be issues with non-default namings
    sample_path  = os.path.join(toppath, series+'_series', series+'_'+sample)
    stages = ['preop', 'postop_1', 'postop_2']
    # mask = skimage.io.imread(os.path.join(sample_path, series+'_'+sample+'_PTL_mask.tif'))
    # mask = mask>0
    # mask = np.transpose(mask, (2,1,0))
    # mask = np.zeros((300,700,1300), dtype = np.uint8)
    # mask[150:220,100:-100,200:-200] = 1
    ims = []
    for stage in stages:
        path = os.path.join(sample_path , series+'_'+sample+'_'+stage, series+'_'+sample+'_'+stage+'_.vol')
        
        # non default naming
        if series+'_'+sample == 'D_1' and stage == 'postop_1':
            path = os.path.join(sample_path , series+'_'+sample+'_postop1', series+'_'+sample+'_postop1_.vol')
        if series+'_'+sample == 'D_1' and stage == 'postop_2':
            path = os.path.join(sample_path , series+'_'+sample+'_postop2', series+'_'+sample+'_postop2_.vol')
            
        im = get_stage_im(path, series, sample, stage)
        ims.append(im)
        
    preopim = ims[0]
    postop1im = ims[1]
    postop2im = ims[2]
    
    #register postop images
    # postop1im = register_images_general(preopim, postop1im)
    # postop2im = register_images_general(preopim, postop2im)
    preopim = register_images_general(postop1im, preopim)
    postop2im = register_images_general(postop1im, postop2im)
    
    #save 16bit images
    outputpath = os.path.join(sample_path, series+'_'+sample+'_'+stages[0]+'_registered.tif')
    skimage.io.imsave(outputpath, np.transpose(preopim,(2,1,0)))
    
    outputpath = os.path.join(sample_path, series+'_'+sample+'_'+stages[1]+'_registered.tif')
    skimage.io.imsave(outputpath, np.transpose(postop1im,(2,1,0)))
    
    outputpath = os.path.join(sample_path, series+'_'+sample+'_'+stages[2]+'_registered.tif')
    skimage.io.imsave(outputpath, np.transpose(postop2im,(2,1,0)))

sample_function('D', '1')

