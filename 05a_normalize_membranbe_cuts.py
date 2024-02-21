# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:15:13 2024

@author: fische_r
"""

import os
import skimage.io
import numpy as np


norm = 0.1

toppath = '/mpc/homes/fische_r/nanotom_data/2023_COELY_postmortem/membrane_cut'

files = os.listdir(toppath)

for file in files:
    path = os.path.join(toppath, file)
    im = skimage.io.imread(path)
    med = np.median(im)
    diff = med - norm
    im = im -diff
    skimage.io.imsave(path, im)
    
    