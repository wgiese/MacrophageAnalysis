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

data_directory = '/media/wgiese/DATA/Projects/Petya/for_wolfgang/'

f = extract_data.ExtractData(data_directory)

df_GFP, df_images_GFP = f.prepare_data(subfolder_name = 'analysis_GFP_all/', key_file = 'overview_GFP_all.xlsx', GFP_flag = True)

df_all, df_images_all = f.prepare_data(subfolder_name = 'analysis_2wk4wk/', key_file = 'overview_2wk_4wk.xlsx', GFP_flag = False)


df_all.to_csv(root_directory + 'output/df_all.csv')
df_GFP.to_csv(root_directory + 'output/df_GFP.csv')



#df_all = pd.read_csv(subfolder_name + 'plots_GFP_all/df_cells.csv') 

#print('df_all')
#print(df_all.columns)

#df_GFP = pd.read_csv(subfolder_name + 'plots_2wk4wk/df_cells.csv') 

#print('df_GFP')
#print(df_GFP.columns)


#df_merged = pd.concat([df_all,df_GFP] , ignore_index = True)

#fig,  axarr = plt.subplots(3, 1, figsize=(5,10), sharey=True)

#sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
#axarr[0].set_ylabel('distance')
#axarr[0].set_title('all vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_thin_vessel', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
#axarr[1].set_ylabel('distance')
#axarr[1].set_title('thin vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_thick_vessel', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2], ci ='sd')
#axarr[2].set_ylabel('distance')
#axarr[2].set_title('thick vessels')

##sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])

##axarr[1].barplot()
#plt.tight_layout()
#plt.savefig('plots_merged/distances.pdf')



#fig,  axarr = plt.subplots(3, 1, figsize=(5,10), sharey=True)

#sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
#axarr[0].set_ylabel('norm. distance')
#axarr[0].set_title('all vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_thin_vessel_norm', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
#axarr[1].set_ylabel('norm. distance')
#axarr[1].set_title('thin vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_thick_vessel_norm', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2], ci ='sd')
#axarr[2].set_ylabel('norm. distance')
#axarr[2].set_title('thick vessels')

##sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])

##axarr[1].barplot()
#plt.tight_layout()
#plt.savefig('plots_merged/normalized_distances.pdf')


#fig,  axarr = plt.subplots(3, 1, figsize=(5,10), sharey=True)

#sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
#axarr[0].set_ylabel('norm. distance')
#axarr[0].set_title('all vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_thin_vessel_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
#axarr[1].set_ylabel('norm. distance')
#axarr[1].set_title('thin vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_thick_vessel_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2], ci ='sd')
#axarr[2].set_ylabel('norm. distance')
#axarr[2].set_title('thick vessels')

##sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])

##axarr[1].barplot()
#plt.tight_layout()
#plt.savefig('plots_merged/normalized_distances_excl_vsl.pdf')


#fig,  axarr = plt.subplots(2,1, figsize=(5,5), sharey=True)

#sns.barplot(data=df_merged , x='tumor_type', y='occupancy', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
#axarr[0].set_ylabel('occupancy')
#axarr[0].set_title('occupancy')

#sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
#axarr[1].set_ylabel('occupancy')
#axarr[1].set_title('occupancy with vessels')


#plt.tight_layout()
#plt.savefig('plots_merged/occupancy.pdf')
