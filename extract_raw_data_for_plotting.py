import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.io import imread, imshow
from skimage.morphology import skeletonize


# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parameter_file", required=True,
	help="provide parameter file! start with ./generate_csv.py -p [name_of_parameter_file]")

args = vars(ap.parse_args())

parameter_file = args["output_directory"]

with open(parameter_file) as f:
    parameters = json.load(f)

data_folder = parameters['']



df_all = pd.read_csv(data_folder + 'df_all.csv') 

print('df_all')
print(df_all.columns)

df_GFP = pd.read_csv(data_folder + 'df_GFP.csv') 

print('df_GFP')
print(df_GFP.columns)


df_merged = pd.concat([df_all,df_GFP] , ignore_index = True)


features = ['mouse_name','MP_type','vessel_file','tumor_type','distance_vessels']


df_raw =  df_merged[features]


df_raw.to_csv(data_folder + 'raw_data.csv')

print(df_raw['mouse_name'].unique())

df_raw_per_mouse = pd.DataFrame()


counter = 0
for mouse in df_raw['mouse_name'].unique():
    df_mouse = df_raw[df_raw['mouse_name']==mouse]
    
    #for tumor_type in df_mouse['tumor_type'].unique():
    for MP_type in df_raw['MP_type'].unique():
        df_MP_type = df_mouse[df_mouse['MP_type']==MP_type]
        
        if(len(df_MP_type)<1):
            continue
        
        df_raw_per_mouse.at[counter, 'mouse_name'] = mouse
        df_raw_per_mouse.at[counter, 'MP_type'] = MP_type
        df_raw_per_mouse.at[counter, 'tumor_type'] = df_MP_type['tumor_type'].unique()[0]
        df_raw_per_mouse.at[counter, 'distance to vessel(mean) [um]'] = df_MP_type['distance_vessels'].mean()
        df_raw_per_mouse.at[counter, 'distance to vessel(SD) [um]'] = df_MP_type['distance_vessels'].std()
        df_raw_per_mouse.at[counter, '#images'] = len(df_MP_type['vessel_file'].unique())
        df_raw_per_mouse.at[counter, '#macrophages'] = len(list(df_MP_type['distance_vessels']))
        
        
        counter += 1
           
    
df_raw_per_mouse.to_csv(data_folder + 'raw_data_per_mouse.csv')
