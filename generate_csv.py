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

root_directory = "/media/wgiese/DATA/Projects/Petya/"
sys.path.append(root_directory)

import MacrophageAnalysis.extract_data as extract_data

data_directory = '/media/wgiese/DATA/Projects/Petya/for_wolfgang_oct19/'

f = extract_data.ExtractData(data_directory)

df_GFP, df_images_GFP = f.prepare_data(subfolder_name = 'analysis_GFP_all/', key_file = 'overview_surv_cells_newdata.xlsx', GFP_flag = True)

df_all, df_images_all = f.prepare_data(subfolder_name = 'analysis_2wk4wk/', key_file = 'overview_2wk_4wk_corrected_size.xlsx', GFP_flag = False)


df_all.to_csv(root_directory + 'output/df_all.csv')
df_GFP.to_csv(root_directory + 'output/df_GFP.csv')

