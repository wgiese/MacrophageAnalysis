import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from joblib import Parallel, delayed
from skimage.measure import label, regionprops
from skimage.io import imread, imshow
from skimage.morphology import skeletonize


data_folder = '/media/wgiese/DATA/Projects/Petya/output/'



df_all = pd.read_csv(data_folder + 'df_all.csv') 

print('df_all')
print(df_all.columns)

df_GFP = pd.read_csv(data_folder + 'df_GFP.csv') 

print('df_GFP')
print(df_GFP.columns)


df_merged = pd.concat([df_all,df_GFP] , ignore_index = True)

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)


fig,  axarr = plt.subplots(3, 1, figsize=(5,10), sharey=True)

sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
axarr[0].set_ylabel('distance')
axarr[0].set_title('all vessels')

sns.barplot(data=df_merged , x='tumor_type', y='distance_thin_vessel', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
axarr[1].set_ylabel('distance')
axarr[1].set_title('thin vessels')

sns.barplot(data=df_merged , x='tumor_type', y='distance_thick_vessel', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2], ci ='sd')
axarr[2].set_ylabel('distance')
axarr[2].set_title('thick vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])

#axarr[1].barplot()
plt.tight_layout()
plt.savefig(data_folder + 'distances.pdf')



fig,  axarr = plt.subplots(3, 1, figsize=(5,10), sharey=True)

sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
axarr[0].set_ylabel('norm. distance')
axarr[0].set_title('all vessels')

sns.barplot(data=df_merged , x='tumor_type', y='distance_thin_vessel_norm', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
axarr[1].set_ylabel('norm. distance')
axarr[1].set_title('thin vessels')

sns.barplot(data=df_merged , x='tumor_type', y='distance_thick_vessel_norm', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2], ci ='sd')
axarr[2].set_ylabel('norm. distance')
axarr[2].set_title('thick vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])

#axarr[1].barplot()
plt.tight_layout()
plt.savefig(data_folder + 'normalized_distances.pdf')


fig,  axarr = plt.subplots(3, 1, figsize=(5,10), sharey=True)

sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0], ci ='sd')
axarr[0].set_ylabel('norm. distance')
axarr[0].set_title('all vessels')

sns.barplot(data=df_merged , x='tumor_type', y='distance_thin_vessel_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1], ci ='sd')
axarr[1].set_ylabel('norm. distance')
axarr[1].set_title('thin vessels')

sns.barplot(data=df_merged , x='tumor_type', y='distance_thick_vessel_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2], ci ='sd')
axarr[2].set_ylabel('norm. distance')
axarr[2].set_title('thick vessels')

#sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels_norm(excl.vsl.)', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])

#axarr[1].barplot()
plt.tight_layout()
plt.savefig(data_folder + 'normalized_distances_excl_vsl.pdf')


fig,  axarr = plt.subplots(5,2, figsize=(20,10), sharey=True)

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r10', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0,0], ci ='sd')
axarr[0,0].set_ylabel('occupancy')
axarr[0,0].set_title('occupancy r = 10 $\mu m$')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r10', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0,1], ci ='sd')
axarr[0,1].set_ylabel('occupancy')
axarr[0,1].set_title('occupancy with vessels r = 10')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r20', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1,0], ci ='sd')
axarr[1,0].set_ylabel('occupancy')
axarr[1,0].set_title('occupancy r = 20 $\mu m$')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r20', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1,1], ci ='sd')
axarr[1,1].set_ylabel('occupancy')
axarr[1,1].set_title('occupancy with vessels r = 20')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r30', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,0], ci ='sd')
axarr[2,0].set_ylabel('occupancy')
axarr[2,0].set_title('occupancy r = 30 $\mu m$')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r30', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,1], ci ='sd')
axarr[2,1].set_ylabel('occupancy')
axarr[2,1].set_title('occupancy with vessels r = 30 $\mu m$')


sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r40', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,0], ci ='sd')
axarr[2,0].set_ylabel('occupancy')
axarr[2,0].set_title('occupancy r = 40 $\mu m$')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r40', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,1], ci ='sd')
axarr[2,1].set_ylabel('occupancy')
axarr[2,1].set_title('occupancy with vessels r = 40 $\mu m$')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r50', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,0], ci ='sd')
axarr[2,0].set_ylabel('occupancy')
axarr[2,0].set_title('occupancy r = 50 $\mu m$')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r50', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,1], ci ='sd')
axarr[2,1].set_ylabel('occupancy')
axarr[2,1].set_title('occupancy with vessels r = 50 $\mu m$')


plt.tight_layout()
plt.savefig(data_folder + 'occupancy.pdf')
