"""
segment CL in postmortem XTM using pretrained weka classifier
runs only within ImageJ
manually open TrainableSegmentation once before running
"""
from ij import IJ #ImageJ version 1.54f
import trainableSegmentation #Trainable Weka Segmentation version 3.3.4
import os
from ij.io import FileSaver


toppath = '/mnt/nas_nanotomData/CT_Data_PSI/FR54/2023_COELY_postmortem'
# wekapath = os.path.join(toppath, 'Weka_segmentation_CL_series_ABCZ')
# datapath = os.path.join(toppath, 'ABCZ_normalized')
# outpath = os.path.join(toppath, 'ABCZ_CL_segmented')

wekapath = os.path.join(toppath, 'Weka_segmentation_CL_series_G')

series = ['G']

for ser in series:
	datapath = os.path.join(toppath, ser+'_normalized')
	outpath = os.path.join(toppath, ser+'_CL_segmented')
	
	if not os.path.exists(outpath):
		os.makedirs(outpath)
	
	files = os.listdir(datapath)
	
	for f in files:
		#load file
		splitfile = f.split('_')
		fileroot = ''.join([i+'_' for i in splitfile[:-1]])
		fileroot = fileroot[:-1]
		if not fileroot[:3] == 'G_5': continue
		print(fileroot)
		im = IJ.openImage(os.path.join(datapath,f))
		
		#segment
		segmentator = trainableSegmentation.WekaSegmentation(im)
		segmentator.loadClassifier(os.path.join(wekapath, 'classifier_G3_preop_postop2.model'))
		segmentator.applyClassifier( 0 )
		result = segmentator.getClassifiedImage()
		
		#save to file
		targetpath = os.path.join(outpath, fileroot+'_ACL_segmented.tif')
		FileSaver(result).saveAsTiff(targetpath)
		IJ.run("Close All")
	
	


