#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from joblib import Parallel, delayed
from skimage.measure import label, regionprops
from skimage.io import imread, imshow
from skimage.morphology import skeletonize


# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parameter_file", required=True,
	help="provide parameter file! start with python generate_csv.py -p [name_of_parameter_file]")
args = vars(ap.parse_args())

parameter_file = args["parameter_file"]

with open(parameter_file) as f:
    parameters = json.load(f)





df_all = pd.read_csv(parameters["output_directory"] + 'df_all.csv') 

print('df_all')
print(df_all.columns)

df_GFP = pd.read_csv(parameters["output_directory"] + 'df_GFP.csv') 

print('df_GFP')
print(df_GFP.columns)


df_merged = pd.concat([df_all,df_GFP] , ignore_index = True)

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
plt.rcParams.update({'font.size': 14})
#sns.set(font_scale=2) 


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
plt.savefig(parameters["output_directory"] + 'distances-thin-thick-all.pdf')
plt.savefig(parameters["output_directory"] + 'distances-thin-thick-all.png')


fig,  ax = plt.subplots(figsize=(10,5))

ax.set_ylim(0,110)

g = sns.barplot(data=df_merged , x='tumor_type', y='distance_vessels', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=ax, ci ='sd', palette = sns.color_palette("Set2"))

sns.despine()

leg = g.axes.get_legend()
leg.set_title("macrophage type")

ax.set_ylabel('distance [$\mathrm{\mu}$m]')
ax.set_xlabel('tumor developement')

#ax.set_title('all vessels')

#plt.xlabel('tumor type')
#plt.xlabel('deviation from flow direction')

plt.tight_layout()
plt.savefig(parameters["output_directory"] + 'distances.pdf')
plt.savefig(parameters["output_directory"] + 'distances.png')

fig,  axarr = plt.subplots(5,1,figsize=(10,20),sharex=True,sharey = True)

costum_bins = np.linspace(0, 150,10)

color_all = "g"
color_implanted= "orange"


df_plot = df_merged[df_merged['tumor_type']=='2 weeks']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[0], color=color_all, kde =True, bins = costum_bins)

df_plot = df_merged[df_merged['tumor_type']=='4 weeks']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[1], color=color_implanted, kde =True, bins = costum_bins)


df_plot = df_merged[df_merged['tumor_type']=='2+2 weeks']
df_plot = df_plot[df_plot['MP_type']=='all']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[2], color=color_all, kde =True, bins = costum_bins)

df_plot = df_merged[df_merged['tumor_type']=='2+2 weeks']
df_plot = df_plot[df_plot['MP_type']=='implanted']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[2], color=color_implanted, kde =True, bins = costum_bins)


df_plot = df_merged[df_merged['tumor_type']=='4+2 weeks']
df_plot = df_plot[df_plot['MP_type']=='all']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[3], color=color_all, kde =True, bins = costum_bins)

df_plot = df_merged[df_merged['tumor_type']=='4+2 weeks']
df_plot = df_plot[df_plot['MP_type']=='implanted']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[3], color=color_implanted, kde =True, bins = costum_bins)



df_plot = df_merged[df_merged['tumor_type']=='4+4 weeks']
df_plot = df_plot[df_plot['MP_type']=='all']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[4], color=color_all, kde =True, bins = costum_bins)

df_plot = df_merged[df_merged['tumor_type']=='4+4 weeks']
df_plot = df_plot[df_plot['MP_type']=='implanted']
g = sns.distplot(df_plot['distance_vessels'], ax=axarr[4], color=color_implanted, kde =True, bins = costum_bins)



axarr[0].set_title('2 weeks')
axarr[1].set_title('4 weeks')
axarr[2].set_title('2+2 weeks')
axarr[3].set_title('4+2 weeks')
axarr[4].set_title('4+4 weeks')

axarr[0].set_xlabel('norm. frequency')
axarr[1].set_xlabel('norm. frequency')
axarr[2].set_xlabel('norm. frequency')
axarr[3].set_xlabel('norm. frequency')
axarr[4].set_xlabel('norm. frequency')


axarr[0].set_ylabel('norm. fre')
axarr[1].set_ylabel('distance [$\mathrm{\mu}$m]')
axarr[2].set_ylabel('distance [$\mathrm{\mu}$m]')
axarr[3].set_ylabel('distance [$\mathrm{\mu}$m]')
axarr[4].set_ylabel('distance [$\mathrm{\mu}$m]')

#sns.despine()

#leg = g.axes.get_legend()
#leg.set_title("macrophage type")

#ax.set_ylabel('distance [$\mathrm{\mu}$m]')
#ax.set_xlabel('tumor developement')

axarr[0].set_xlim(0,150)

plt.tight_layout()
plt.savefig(parameters["output_directory"] + 'distances_distribution.pdf')
plt.savefig(parameters["output_directory"] + 'distances_distribution.png')



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
plt.savefig(parameters["output_directory"] + 'normalized_distances.pdf')
plt.savefig(parameters["output_directory"] + 'normalized_distances.png')


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
plt.savefig(parameters["output_directory"] + 'normalized_distances_excl_vsl.pdf')
plt.savefig(parameters["output_directory"] + 'normalized_distances_excl_vsl.png')

'''
plot nearest neighbour metric
'''

fig,  axarr = plt.subplots(2,1, figsize=(20,10), sharey=True)

sns.barplot(data=df_merged , x='tumor_type', y='nearest_neighbour', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0])
axarr[0].set_ylabel('dist. in $\mu m$')
axarr[0].set_title('nearest neighbour')


sns.barplot(data=df_merged , x='tumor_type', y='4nearest_neighbour', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1])
axarr[1].set_ylabel('dist. in $\mu m$')
axarr[1].set_title('4 nearest neighbours')

plt.tight_layout()
plt.savefig(parameters["output_directory"] + 'nearest_neighbour_bar.pdf')
plt.savefig(parameters["output_directory"] + 'nearest_neighbour_bar.png')



fig,  axarr = plt.subplots(5,2, figsize=(20,10), sharey=True)

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r10', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0,0])
axarr[0,0].set_ylabel('occupancy')
axarr[0,0].set_title('occupancy (r = 10 $\mu m$)')
axarr[0,0].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r10', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0,1])
axarr[0,1].set_ylabel('occupancy')
axarr[0,1].set_title('occupancy with vessels (r = 10 $\mu m$)')
axarr[0,1].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r20', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1,0])
axarr[1,0].set_ylabel('occupancy')
axarr[1,0].set_title('occupancy (r = 20 $\mu m$)')
axarr[1,0].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r20', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[1,1])
axarr[1,1].set_ylabel('occupancy')
axarr[1,1].set_title('occupancy with vessels (r = 20 $\mu m$)')
axarr[1,1].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r30', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,0])
axarr[2,0].set_ylabel('occupancy')
axarr[2,0].set_title('occupancy (r = 30 $\mu m$)')
axarr[2,0].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r30', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[2,1])
axarr[2,1].set_ylabel('occupancy')
axarr[2,1].set_title('occupancy with vessels (r = 30 $\mu m$)')
axarr[2,1].axhline(1.0, ls='--')


sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r40', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[3,0])
axarr[3,0].set_ylabel('occupancy')
axarr[3,0].set_title('occupancy (r = 40 $\mu m$)')
axarr[3,0].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r40', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[3,1])
axarr[3,1].set_ylabel('occupancy')
axarr[3,1].set_title('occupancy with vessels (r = 40 $\mu m$)')
axarr[3,1].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_r50', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[4,0])
axarr[4,0].set_ylabel('occupancy')
axarr[4,0].set_title('occupancy (r = 50 $\mu m$)')
axarr[4,0].axhline(1.0, ls='--')

sns.barplot(data=df_merged , x='tumor_type', y='occupancy_with_vessels_r50', hue = "MP_type", order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[4,1])
axarr[4,1].set_ylabel('occupancy')
axarr[4,1].set_title('occupancy with vessels (r = 50 $\mu m$)')
axarr[4,1].axhline(1.0, ls='--')

plt.tight_layout()
plt.savefig(parameters["output_directory"] + 'occupancy.pdf')
plt.savefig(parameters["output_directory"] + 'occupancy.png')
