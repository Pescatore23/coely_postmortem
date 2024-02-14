"""
segment CL in postmortem XTM using pretrained weka classifier
runs only within ImageJ
"""
from ij import IJ
import trainableSegmentation
import os
from ij.io import FileSaver


toppath = '/mnt/nas_nanotomData/CT_Data_PSI/FR54/2023_COELY_postmortem'
wekapath = os.path.join(toppath, 'Weka_segmentation_CL_series_ABCZ')
datapath = os.path.join(toppath, 'ABCZ_normalized')
outpath = os.path.join(toppath, 'ABCZ_CL_segmented')
if not os.path.exists(outpath):
	os.makedirs(outpath)

files = os.listdir(datapath)

segmentator = trainableSegmentation.WekaSegmentation()
segmentator.loadClassifier(os.path.join(wekapath, 'classifier.model')

for f in files:
	splitfile = f.split('_')
	fileroot = ''.join([i+'_' for i in splitfile[:-1])
	fileroot = fileroot[:-1]
	print(fileroot)
	im = IJ.openImage(os.path.join(datapath,f)
	result = segmentator.applyClassifier(im, 0, 0) #(image to be segmented, number of threats (0=all), result type); result type 0 for labeled image, 1 for probability map of each phase (=hyperstack)
	
	targetpath = os.path.join(outpath, fileroot+'_ACL_segmented.tif')
	FileSaver(result).saveAsTiff(targetpath)
	im.close()
	
	


