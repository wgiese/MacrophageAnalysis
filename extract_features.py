#!/usr/bin/env python3

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
import data_processing

 
# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parameter_file", required=True,
	help="provide parameter file! start with ./generate_csv.py -p [name_of_parameter_file]")

args = vars(ap.parse_args())

parameter_file = args["parameter_file"]

with open(parameter_file) as f:
    parameters = json.load(f)

f = data_processing.ExtractData(parameters)

df, df_images = f.prepare_data()

df.to_csv(parameters["output_directory"] + "raw_results_" +"_".join(parameters["macrophage_channel"]) + ".csv")


'''
extract selected features per macrophage and per mouse
'''


columns = ['mouse_name','MP_type','vessel_file','tumor_type',parameters["selected_feature"]]
df_selected =  df[columns]


df_selected.to_csv(parameters["output_directory"] + parameters["selected_feature"] + "_per_macrophage"+ "_".join(parameters["macrophage_channel"]) + ".csv")
df_selected_per_mouse = pd.DataFrame()


counter = 0
for mouse in df_selected['mouse_name'].unique():
    df_mouse = df_selected[df_selected['mouse_name']==mouse]
    
    for MP_type in df_raw['MP_type'].unique():
        df_MP_type = df_mouse[df_mouse['MP_type']==MP_type]
        
        if(len(df_MP_type)<1):
            continue
        
        df_selected_per_mouse.at[counter, 'mouse_name'] = mouse
        df_selected_per_mouse.at[counter, 'MP_type'] = MP_type
        df_selected_per_mouse.at[counter, 'tumor_type'] = df_MP_type['tumor_type'].unique()[0]
        df_selected_per_mouse.at[counter, 'distance to vessel(mean) [um]'] = df_MP_type['distance_vessels'].mean()
        df_selected_per_mouse.at[counter, 'distance to vessel(SD) [um]'] = df_MP_type['distance_vessels'].std()
        df_selected_per_mouse.at[counter, '#images'] = len(df_MP_type['vessel_file'].unique())
        df_selected_per_mouse.at[counter, '#macrophages'] = len(list(df_MP_type['distance_vessels']))
      
        counter += 1
        
    
df_selected_per_mouse.to_csv(parameters["output_directory"] + parameters["selected_feature"] + "_per_mouse"+ "_".join(parameters["macrophage_channel"]) + ".csv")

