import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage.morphology import distance_transform_edt
from joblib import Parallel, delayed
from skimage.measure import label, regionprops
from skimage.io import imread, imshow
from skimage.morphology import skeletonize


class ExtractData:
    """
    In the constructor give folder from which to extract     
    import with:
    import MacrophageAnalysis.data_loader as data_loader
    """

    def __init__(self, rootfolder_name_):

        self.rootfolder_name = rootfolder_name_


    def calculate_occupancy(self, MP_pos, vessel_im_dist_transform, radius_mum = 10.0, pixel_per_mum = 1024.0/445.0 ):
        
                    
        '''
        calculate occupancy
        '''
        
        occupancy_radius_pixel = int(radius_mum*pixel_per_mum)
        sample_points = 10000
        
        x_ext = np.array(vessel_im_dist_transform).shape[0]-1
        y_ext = np.array(vessel_im_dist_transform).shape[1]-1
        

        #occupied_image = np.zeros((x_ext,y_ext))
        
        
        
        #for ix,iy in np.ndindex(occupied_image.shape):
            
            #for index, row in MP_pos.iterrows():
                #MP_x = int(row['X'])
                #MP_y = int(row['Y'])
                
                #dist2 = (MP_x - ix)*(MP_x - ix) + (MP_y - iy)*(MP_y - iy)
                
                #if dist2 < occupancy_radius_pixel*occupancy_radius_pixel:
                    #occupied_image[ix,iy] = 1
                    
        #area1 = np.count_nonzero(occupied_image)  
        #area2 = np.pi*float(occupancy_radius_pixel*occupancy_radius_pixel)*float(len(MP_pos['X']))
            
        #occupancy =  area1/area2
            
        #if(occupancy > 1.0):
            #print("================================")

            #print("something is wrong with the calculation!")
            #print("There are %s macrophages" % len(MP_pos['X']))
            #print("shape of the image: (%s,%s)" % (x_ext,y_ext))
            #print("area1: %s" % area1)
            #print("area2: %s" % area2)
            #print("occupancy radius (pixel): %s" % occupancy_radius_pixel)
            #print("occupancy: %s" % occupancy)
            #print("================================")
        #else:            
            #print("occupancy smaller 1! ")
            
            
        points_x = np.random.randint(low=1, high=x_ext, size=sample_points)
        points_y = np.random.randint(low=1, high=y_ext, size=sample_points)
        
        list_occupancy = []
        
        for index, row in MP_pos.iterrows():
            
            MP_x = int(row['X'])
            MP_y = int(row['Y'])
            
            for k in range(sample_points):
                
                p_x = points_x[k]
                p_y = points_y[k]
                
                dist2 = (MP_x - p_x)*(MP_x - p_x) + (MP_y - p_y)*(MP_y - p_y)
            
                if (dist2 < occupancy_radius_pixel*occupancy_radius_pixel):
                    list_occupancy.append(k)
            
        points_occupied = len(set(list_occupancy))
        fractional_area = float(points_occupied)/float(sample_points)
        total_area = float(x_ext*y_ext)
            
        area1 = total_area*fractional_area
        area2 = np.pi*float(occupancy_radius_pixel*occupancy_radius_pixel)*float(len(MP_pos['X']))
        
        occupancy =  area1/area2
        
        if(occupancy > 1.0):
            print("================================")
            #print("points_x")
            #print(points_x)
            #print("MP_pos")
            #print(MP_pos['X'])
            #print(list_occupancy)
            print("fractional_area: %s" % fractional_area)
            print("something is wrong with the calculation!")
            print("%s out of %s points are occupied" % (len(set(list_occupancy)), sample_points) )
            print("There are %s macrophages" % len(MP_pos['X']))
            print("shape of the image: (%s,%s)" % (x_ext,y_ext))
            print("area1: %s" % area1)
            print("area2: %s" % area2)
            print("occupancy radius (pixel): %s" % occupancy_radius_pixel)
            print("occupancy: %s" % occupancy)
            print("================================")
        else:            
            print("occupancy smaller 1! ")
        
        '''
        calculate occupancy with vessels
        '''
        
    
        list_occupancy_with_vessels = []
        
        for index, row in MP_pos.iterrows():
            
            #for p_x,p_y in zip(points_x, points_y):
            for k in range(len(points_x)):
                
                p_x = points_x[k]
                p_y = points_y[k]
                
                dist2 = (row['X'] - p_x)**2 + (row['Y'] - p_y)**2
            
                if (dist2 < occupancy_radius_pixel*occupancy_radius_pixel):
                    list_occupancy_with_vessels.append(k)
                    
                if vessel_im_dist_transform[ p_x, p_y] < occupancy_radius_pixel:
                    list_occupancy_with_vessels.append(k)
                
                
        area1 = float(x_ext*y_ext*len(set(list_occupancy_with_vessels)))/float(sample_points)
        #area2 = float(np.pi*occupancy_radius_pixel*occupancy_radius_pixel*len(MP_pos['X']) + np.count_nonzero(vessel_im_dist_transform))
         
        area2 = float(np.pi*occupancy_radius_pixel*occupancy_radius_pixel*len(MP_pos['X']) + (vessel_im_dist_transform < occupancy_radius_pixel).sum())
         
         
        occupancy_with_vessels = area1/area2
        
        return occupancy, occupancy_with_vessels
    
    
    
    
    def prepare_data(self, subfolder_name = 'analysis_GFP_all/', key_file = 'overview_GFP_all.xlsx', GFP_flag = True):
        
        
        overview_file = pd.read_excel(self.rootfolder_name + key_file)
        
        if (GFP_flag == False):
            overview_file = overview_file.rename(columns={"analysis_file_macrophages": "analysis_file_macrophages_all", "genotype" : "tumor_type"})
        
        subfolder_name = self.rootfolder_name + subfolder_name
        
        print(overview_file.columns)
        
        overview_file  = overview_file.drop([0,1,20])

        overview_file['analysis_file_vessels'] = [subfolder_name + fn_['analysis_file_vessels'] for _, fn_ in
                                                overview_file.iterrows()]
        overview_file['analysis_file_macrophages_all'] = [subfolder_name + fn_['analysis_file_macrophages_all'] for _, fn_ in
                                                    overview_file.iterrows()]
        
        if (GFP_flag):
            overview_file['analysis_file_macrophages_GFP'] = [subfolder_name + fn_['analysis_file_macrophages_GFP'] for _, fn_ in
                                                        overview_file.iterrows()]

        df_images = pd.DataFrame()
        df_images.index.name = 'vessel_file'
        df_cells = pd.DataFrame()

        width_threshold_um=10

        for _, fn_ in overview_file.iterrows():
            MP_pos = pd.read_csv(fn_['analysis_file_macrophages_all'])
            vessel_im_all = (plt.imread(fn_['analysis_file_vessels']) > 0).astype(int)
            
            
            print("shape of vessel image")
            print(np.array(vessel_im_all).shape)

            # identify thin and thick vessels
            vessel_label = label(vessel_im_all == 0)

            rp = regionprops(vessel_label, intensity_image=skeletonize(vessel_im_all == 0).astype('uint8'))
            rp_df = pd.DataFrame([{'analysis_file_vessels': fn_['analysis_file_vessels'], 'area': r['area'], 'width': 1. / r['mean_intensity'],
                                'minor_axis_length': r['minor_axis_length']} for r in rp])

            vessel_widths_1 = np.zeros_like(vessel_label)
            vessel_widths_2 = np.zeros_like(vessel_label)

            # vessel_skel = skeletonize(vessel_im_all == 0).astype('uint8')

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
            
            occupancy_r10, occupancy_with_vessels_r10 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 10.0)
            occupancy_r20, occupancy_with_vessels_r20 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 20.0)
            occupancy_r30, occupancy_with_vessels_r30 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 30.0)
            occupancy_r40, occupancy_with_vessels_r40 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 40.0)
            occupancy_r50, occupancy_with_vessels_r50 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 50.0)
            

            df_cells = df_cells.append(pd.DataFrame({'distance_vessels': distances,
                                            'distance_thin_vessel': distances_thin,
                                            'distance_thick_vessel': distances_thick,
                                            'distance_vessels_norm': distances_norm,
                                            'distance_thin_vessel_norm': distances_thin_norm,
                                            'distance_thick_vessel_norm': distances_thick_norm,
                                            'distance_vessels_norm(excl.vsl.)': distances_norm_excl_vsl,
                                            'distance_thin_vessel_norm(excl.vsl.)': distances_thin_norm_excl_vsl,
                                            'distance_thick_vessel_norm(excl.vsl.)': distances_thick_norm_excl_vsl,
                                            'occupancy_r10' : occupancy_r10,
                                            'occupancy_with_vessels_r10' : occupancy_with_vessels_r10,
                                            'occupancy_r20' : occupancy_r20,
                                            'occupancy_with_vessels_r20' : occupancy_with_vessels_r20,
                                            'occupancy_r30' : occupancy_r30,
                                            'occupancy_with_vessels_r30' : occupancy_with_vessels_r30,
                                            'occupancy_r40' : occupancy_r40,
                                            'occupancy_with_vessels_r40' : occupancy_with_vessels_r40,
                                            'occupancy_r50' : occupancy_r50,
                                            'occupancy_with_vessels_r50' : occupancy_with_vessels_r50,
                                            'vessel_file': fn_['analysis_file_vessels'],
                                            'vessel_density': 1 - np.sum(vessel_im_all) / (np.shape(vessel_im_all)[0] * np.shape(vessel_im_all)[1]),
                                            'tumor_type': fn_['tumor_type'],
                                            'MP_type': 'all',
                                            'mouse_name': str(fn_['mouse_name'])}))
            
            if not GFP_flag:
                continue
            
            if (not fn_['analysis_file_macrophages_GFP'].endswith('None')):
                print(fn_['analysis_file_macrophages_GFP'])
                MP_pos_GFP = pd.read_csv(fn_['analysis_file_macrophages_GFP'])
                distances_GFP = vessel_im_dist_transform[MP_pos_GFP['X'], MP_pos_GFP['Y']]
                distances_thin_GFP = vessel_im_dist_thin_transform[MP_pos_GFP['X'], MP_pos_GFP['Y']]
                distances_thick_GFP = vessel_im_dist_thick_transform[MP_pos_GFP['X'], MP_pos_GFP['Y']]

                distances_GFP_norm = distances_GFP/np.mean(vessel_im_dist_transform)
                distances_thin_GFP_norm = distances_thin_GFP/np.mean(vessel_im_dist_thin_transform)
                distances_thick_GFP_norm = distances_thick_GFP/np.mean(vessel_im_dist_thick_transform)
                
                norm_value = float(np.sum(vessel_im_dist_transform))/float(np.count_nonzero(vessel_im_dist_transform))
                
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
                
                
                #norm_thin_value = float(np.sum(vessel_im_dist_thin_transform))/float(np.count_nonzero(vessel_im_dist_transform))
                #norm_thick_value = float(np.sum(vessel_im_dist_thick_transform))/float(np.count_nonzero(vessel_im_dist_transform))
            
                distances_GFP_norm_excl_vsl = distances_GFP/norm_value
                distances_thin_GFP_norm_excl_vsl = distances_thin_GFP/norm_thin_value
                distances_thick_GFP_norm_excl_vsl = distances_thick_GFP/norm_thick_value 
                
                
                occupancy_r10, occupancy_with_vessels_r10 = self.calculate_occupancy(MP_pos_GFP,vessel_im_dist_transform, radius_mum = 10.0)
                occupancy_r20, occupancy_with_vessels_r20 = self.calculate_occupancy(MP_pos_GFP,vessel_im_dist_transform, radius_mum = 20.0)
                occupancy_r30, occupancy_with_vessels_r30 = self.calculate_occupancy(MP_pos_GFP,vessel_im_dist_transform, radius_mum = 30.0)
                occupancy_r40, occupancy_with_vessels_r40 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 40.0)
                occupancy_r50, occupancy_with_vessels_r50 = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, radius_mum = 50.0)

                df_cells = df_cells.append(pd.DataFrame({'distance_vessels': distances_GFP,
                                                'distance_thin_vessel': distances_thin_GFP,
                                                'distance_thick_vessel': distances_thick_GFP,
                                                'distance_vessels_norm': distances_GFP_norm,
                                                'distance_thin_vessel_norm': distances_thin_GFP_norm,
                                                'distance_thick_vessel_norm': distances_thick_GFP_norm,
                                                'distance_vessels_norm(excl.vsl.)': distances_GFP_norm_excl_vsl,
                                                'distance_thin_vessel_norm(excl.vsl.)': distances_thin_GFP_norm_excl_vsl,
                                                'distance_thick_vessel_norm(excl.vsl.)': distances_thick_GFP_norm_excl_vsl,
                                                'occupancy_r10' : occupancy_r10,
                                                'occupancy_with_vessels_r10' : occupancy_with_vessels_r10,
                                                'occupancy_r20' : occupancy_r20,
                                                'occupancy_with_vessels_r20' : occupancy_with_vessels_r20,
                                                'occupancy_r30' : occupancy_r30,
                                                'occupancy_with_vessels_r30' : occupancy_with_vessels_r30,
                                                'occupancy_r40' : occupancy_r40,
                                                'occupancy_with_vessels_r40' : occupancy_with_vessels_r40,
                                                'occupancy_r50' : occupancy_r50,
                                                'occupancy_with_vessels_r50' : occupancy_with_vessels_r50,
                                                'vessel_file': fn_['analysis_file_vessels'],
                                                'vessel_density': 1 - np.sum(vessel_im_all) / (np.shape(vessel_im_all)[0] * np.shape(vessel_im_all)[1]),
                                                'tumor_type': fn_['tumor_type'],
                                                'MP_type': 'implanted',
                                                'mouse_name': str(fn_['mouse_name'])}))

                df_images = df_images.append(pd.DataFrame({'vessel_number': vessel_number,
                            'vessel_number_thin': vessel_number_thin,
                            'vessel_number_thick': vessel_number_thick,
                            # 'vessel_props': rp_df,
                            'vessel_width_im': [vessel_widths_1],
                            'vessel_width_im_thin': [vessel_im_thin],
                            'vessel_width_im_thick': [vessel_im_thick],
                            'vessel_minor_axis_im': [vessel_widths_2],
                            'tumor_type': fn_['tumor_type'],
                            'distance': np.mean(distances),
                            'distance_thin': np.mean(distances_thin),
                            'distance_thick': np.mean(distances_thick),
                            'distance_GFP': np.mean(distances_GFP),
                            'distance_thin_GFP': np.mean(distances_thin_GFP),
                            'distance_thick_GFP': np.mean(distances_thick_GFP),
                            'distance_norm': np.mean(distances_norm),
                            'distance_thin_norm': np.mean(distances_thin_norm),
                            'distance_thick_norm': np.mean(distances_thick_norm),
                            'distance_GFP_norm': np.mean(distances_GFP_norm),
                            'distance_thin_GFP_norm': np.mean(distances_thin_GFP_norm),
                            'distance_thick_GFP_norm': np.mean(distances_thick_GFP),
                            'mouse_name': str(fn_['mouse_name'])}, index=[fn_['analysis_file_vessels']]))
            else:
                df_images = df_images.append(pd.DataFrame({'vessel_number': vessel_number,
                    'vessel_number_thin': vessel_number_thin,
                    'vessel_number_thick': vessel_number_thick,
                    'vessel_width_im': [vessel_widths_1],
                    'vessel_width_im_thin': [vessel_im_thin],
                    'vessel_width_im_thick': [vessel_im_thick],
                    'vessel_minor_axis_im': [vessel_widths_2],
                    'tumor_type': fn_['tumor_type'],
                    'distance': np.mean(distances),
                    'distance_thin': np.mean(distances_thin),
                    'distance_thick': np.mean(distances_thick),
                    'mouse_name': str(fn_['mouse_name'])}, index=[fn_['analysis_file_vessels']]))

        df_cells['mouse_name_MP'] = [str(n_['mouse_name']) + ', ' + n_['MP_type'] for _, n_ in df_cells.iterrows()]

        return df_cells, df_images
