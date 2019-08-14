import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from joblib import Parallel, delayed
from skimage.measure import label, regionprops
from skimage.io import imread, imshow
from skimage.morphology import skeletonize

def prepare_data():
    overview_file = pd.read_excel('/media/wgiese/DATA/Projects/Petya/for_wolfgang/overview_2wk_4wk.xlsx')
    #overview_file = pd.read_csv('overview_short.csv')
    # complete filename
    
    overview_file  = overview_file.drop([0])
    
    
    
    #for _, fn_ in overview_file.iterrows():
        #print("-----")
        #print("index: %s" % _)
        #print(fn_['analysis_file_vessels'])
    
    
    subfolder_name = '/media/wgiese/DATA/Projects/Petya/for_wolfgang/analysis_2wk4wk/'

    overview_file['analysis_file_vessels'] = [subfolder_name + fn_['analysis_file_vessels'] for _, fn_ in
                                              overview_file.iterrows()]
    overview_file['analysis_file_macrophage'] = [subfolder_name + fn_['analysis_file_macrophages'] for _, fn_ in
                                                  overview_file.iterrows()]

    df_images = pd.DataFrame()
    df_images.index.name = 'vessel_file'
    df_cells = pd.DataFrame()

    width_threshold_um=10

    for _, fn_ in overview_file.iterrows():
        
        print("Analysis file: %s" % fn_['analysis_file_vessels'])
        print("%s" % fn_['analysis_file_macrophage'])
        
        MP_pos = pd.read_csv(fn_['analysis_file_macrophage'])
        vessel_im_all = (plt.imread(fn_['analysis_file_vessels']) > 0).astype(int)
        
        #print(vessel_im_all[vessel_im_all == 0]) 

        # identify thin and thick vessels
        vessel_label = label(vessel_im_all == 0)

        rp = regionprops(vessel_label, intensity_image=skeletonize(vessel_im_all == 0).astype('uint8'))
        rp_df = pd.DataFrame([{'analysis_file_vessels': fn_['analysis_file_vessels'], 'area': r['area'], 'width': 1. / r['mean_intensity'],
                               'minor_axis_length': r['minor_axis_length']} for r in rp])
        
        #print(rp_df)

        vessel_widths_1 = np.zeros_like(vessel_label)
        vessel_widths_2 = np.zeros_like(vessel_label)
        
        for l in np.unique(vessel_label):
            if l==0:
                continue
            vessel_widths_1[vessel_label == l] = np.min(np.array([1. / rp[l - 1]['mean_intensity'], 2*np.sqrt(rp[l-1]['area']/np.pi)]))*(fn_['x_resolution [um]']/1023)
            vessel_widths_2[vessel_label == l] = rp[l - 1]['minor_axis_length']*(fn_['x_resolution [um]']/1023)

        vessel_im_thin = np.array(vessel_widths_1 > 0) * np.array(vessel_widths_1 <= width_threshold_um) * vessel_widths_1
        vessel_im_thick = np.array(vessel_widths_1 > 0) * np.array(vessel_widths_1 > width_threshold_um) * vessel_widths_1

        # use distance transform to obtain distances of all macrophages from the vessels
        vessel_im_dist_transform = distance_transform_edt(vessel_im_all)*fn_['x_resolution [um]']/1023
        vessel_im_dist_thin_transform = distance_transform_edt((vessel_im_thin==0).astype(int))*fn_['x_resolution [um]']/1023
        vessel_im_dist_thick_transform = distance_transform_edt((vessel_im_thick==0).astype(int))*fn_['x_resolution [um]']/1023

        vessel_number = len(np.unique(vessel_widths_1))-1
        vessel_number_thin = len(np.unique(vessel_im_thin))-1
        vessel_number_thick = len(np.unique(vessel_im_thick))-1

        distances = vessel_im_dist_transform[MP_pos['X'], MP_pos['Y']]
        distances_thin = vessel_im_dist_thin_transform[MP_pos['X'], MP_pos['Y']]
        distances_thick = vessel_im_dist_thick_transform[MP_pos['X'], MP_pos['Y']]
        distances_norm = distances/np.mean(vessel_im_dist_transform)
        distances_thin_norm = distances_thin/np.mean(vessel_im_dist_thin_transform)
        distances_thick_norm = distances_thick/np.mean(vessel_im_dist_thick_transform)
        
        norm_value = float(np.sum(vessel_im_dist_transform))/float(np.count_nonzero(vessel_im_dist_transform))
        #norm_thin_value = float(np.sum(vessel_im_dist_thin_transform))/float(np.count_nonzero(vessel_im_dist_transform))
        #norm_thick_value = float(np.sum(vessel_im_dist_thick_transform))/float(np.count_nonzero(vessel_im_dist_transform))
        
        distances_norm_excl_vsl = distances/norm_value
                
        norm_thin_value = 0.0
        count_dist_thin = 0.0
        
        for ix,iy in np.ndindex(vessel_im_dist_thin_transform.shape):
            if (vessel_im_dist_transform[ix,iy] > 0):
                count_dist_thin += 1.0
                norm_thin_value += float(vessel_im_dist_thin_transform[ix,iy])
                
        norm_thin_value = norm_thin_value/count_dist_thin
        
        norm_thick_value = 0.0
        count_dist_thick = 0.0
        
        for ix,iy in np.ndindex(vessel_im_dist_thick_transform.shape):
            if (vessel_im_dist_transform[ix,iy] > 0):
                count_dist_thick += 1.0
                norm_thick_value += float(vessel_im_dist_thick_transform[ix,iy])
                
        norm_thick_value = norm_thick_value/count_dist_thick
        
        
        distances_thin_norm_excl_vsl = distances_thin/norm_thin_value
        distances_thick_norm_excl_vsl = distances_thick/norm_thick_value 
        
        
        '''
        calculate Moran's I
        '''
        
        #mean_val = len(MP_pos)/float(1024*1024 - np.array(vessel_im_all).sum())
        #moransI = 0.0
        
        #MP_image = np.zeros((1024,1024))
        
        #for index, row in MP_pos.iterrows():
            #MP_image[MP_pos['X'],MP_pos['Y']] = 1.0
            
        #mean_val = MP_image.mean()
        
        #for ix,iy in np.ndindex(MP_image.shape):
            #dist = ix
            #moransI += np.exp()
            
            
            
        '''
        calculate occupancy
        '''
        
        occupancy_radius = 50.0
        sample_points = 1000
        
        points_x = np.random.randint(low=1, high=1024, size=sample_points)
        points_y = np.random.randint(low=1, high=1024, size=sample_points)
        
        list_occupancy = []
        
        for index, row in MP_pos.iterrows():
            
            #for p_x,p_y in zip(points_x, points_y):
            for k in range(len(points_x)):
                
                p_x = points_x[k]
                p_y = points_y[k]
                
                dist2 = (row['X'] - p_x)**2 + (row['Y'] - p_y)**2
            
                if (dist2 < occupancy_radius*occupancy_radius):
                    list_occupancy.append(k)
              
        area1 = float(1024*1024*len(set(list_occupancy)))/float(sample_points)
        area2 = float(np.pi*occupancy_radius*occupancy_radius*len(MP_pos['X']))
        
        occupancy =  area1/area2
        
        '''
        calculate occupancy with vessels
        '''
        
        occupancy_radius = 50.0
            
        list_occupancy_with_vessels = []
        
        for index, row in MP_pos.iterrows():
            
            #for p_x,p_y in zip(points_x, points_y):
            for k in range(len(points_x)):
                
                p_x = points_x[k]
                p_y = points_y[k]
                
                dist2 = (row['X'] - p_x)**2 + (row['Y'] - p_y)**2
            
                if (dist2 < occupancy_radius*occupancy_radius):
                    list_occupancy_with_vessels.append(k)
                    
                if vessel_im_dist_transform[ p_x, p_y] == 0:
                    list_occupancy_with_vessels.append(k)
                
                
        area1 = float(1024*1024*len(set(list_occupancy_with_vessels)))/float(sample_points)
        area2 = float(np.pi*occupancy_radius*occupancy_radius*len(MP_pos['X']) +  np.count_nonzero(vessel_im_dist_transform))
                
        occupancy_with_vessels = area1/area2
                   
        

        df_cells = df_cells.append(pd.DataFrame({'distance_vessels': distances,
                                           'distance_thin_vessel': distances_thin,
                                           'distance_thick_vessel': distances_thick,
                                           'distance_vessels_norm': distances_norm,
                                           'distance_thin_vessel_norm': distances_thin_norm,
                                           'distance_thick_vessel_norm': distances_thick_norm,
                                           'distance_vessels_norm(excl.vsl.)': distances_norm_excl_vsl,
                                           'distance_thin_vessel_norm(excl.vsl.)': distances_thin_norm_excl_vsl,
                                           'distance_thick_vessel_norm(excl.vsl.)': distances_thick_norm_excl_vsl,
                                           'occupancy' : occupancy,
                                           'occupancy_with_vessels' : occupancy_with_vessels,
                                           'vessel_file': fn_['analysis_file_vessels'],
                                           'vessel_density': 1 - np.sum(vessel_im_all) / (np.shape(vessel_im_all)[0] * np.shape(vessel_im_all)[1]),
                                           'MP_type': 'all',
                                           'tumor_type' : fn_['genotype'],
                                           'mouse_name': str(fn_['mouse_name'])}))

        df_images = df_images.append(pd.DataFrame({'vessel_number': vessel_number,
            'vessel_number_thin': vessel_number_thin,
            'vessel_number_thick': vessel_number_thick,
            'vessel_width_im': [vessel_widths_1],
            'vessel_width_im_thin': [vessel_im_thin],
            'vessel_width_im_thick': [vessel_im_thick],
            'vessel_minor_axis_im': [vessel_widths_2],
            #'tumor_type': fn_['tumor_type'],
            'distance': np.mean(distances),
            'distance_thin': np.mean(distances_thin),
            'distance_thick': np.mean(distances_thick),
            'mouse_name': str(fn_['mouse_name'])}, index=[fn_['analysis_file_vessels']]))

    df_cells['mouse_name_MP'] = [str(n_['mouse_name']) + ', ' + n_['MP_type'] for _, n_ in df_cells.iterrows()]

    return df_cells, df_images

    # plt.figure()
    # imshow(all_df.iloc[0]['vessel_width_im'])
    # plt.colorbar()

df_cells, df_images = prepare_data()
# df_cells['is_GFP'] = [n_['MP_type']=='implanted' for _, n_ in df_cells.iterrows()]

print("==================================")
print(df_cells)
#print("==================================")
#print(df_images)

df_cells.to_csv('plots_2wk4wk/df_cells.csv')

#fig,  axarr = plt.subplots(3, 1, sharey=True)

fig,  axarr = plt.subplots(2, 1, sharey=True)

sns.barplot(data=df_cells, x='tumor_type', y='distance_vessels_norm', ax=axarr[0])#order=('2 weeks', '4 weeks', '2+2 weeks', '4+2 weeks', '4+4 weeks'), ax=axarr[0])

sns.barplot(data=df_cells, x='tumor_type', y='distance_vessels_norm(excl.vsl.)', ax=axarr[1])

#axarr[1].barplot()
plt.tight_layout()
plt.savefig('plots_2wk4wk//plots.pdf')

print(df_cells['distance_vessels_norm'].mean())
print(df_cells['distance_vessels_norm(excl.vsl.)'].mean())

