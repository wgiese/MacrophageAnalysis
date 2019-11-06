import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from joblib import Parallel, delayed
from skimage.measure import label, regionprops
from skimage.io import imread, imshow
from skimage.morphology import skeletonize

import json
import argparse
import extract_data

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parameter_file", required=True,
	help="provide parameter file! start with python generate_csv.py -p [name_of_parameter_file]")
args = vars(ap.parse_args())

parameter_file = args["parameter_file"]

with open(parameter_file) as f:
    d = json.load(f)

f = extract_data.ExtractData(data_directory)

df_GFP, df_images_GFP = f.prepare_data(subfolder_name = d["subfolder_vessel_images_2"], key_file = d["meta_data_file_2"] , GFP_flag = True)
df_all, df_images_all = f.prepare_data(subfolder_name = d["subfolder_vessel_images_1"], key_file = d["meta_data_file_1"], GFP_flag = False)


#df_all.to_csv(root_directory + 'output/df_all.csv')
#df_GFP.to_csv(root_directory + 'output/df_GFP.csv')

