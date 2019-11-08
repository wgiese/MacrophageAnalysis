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
    import data_processing
    """

    def __init__(self, parameters):

        self.parameters = parameters
        self.rootfolder_name = parameters["data_directory"]


    def calculate_occupancy(self, MP_pos, vessel_im_dist_transform, radius_mum, pixel_per_mum):
        
                    
        '''
        calculate occupancy
        '''
        
        occupancy_radius_pixel = int(radius_mum*pixel_per_mum)
        sample_points = self.parameters['occupancy_sampling']
        
        x_ext = np.array(vessel_im_dist_transform).shape[0]-1
        y_ext = np.array(vessel_im_dist_transform).shape[1]-1
        
           
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
    
    
    def calculate_nearest_neighbours(self, MP_pos, pixel_per_mum):
        
        dists_nn = []
        dists_4nn = []
        
        for index_out, row_out in MP_pos.iterrows():
            
            dists = []
            
            for index_in, row_in in MP_pos.iterrows():
                
                if (index_in != index_out):
                    dist2 = (row_out['X'] - row_in['X'])**2 + (row_out['X'] - row_in['X'])**2
                    dists.append(np.sqrt(dist2))
                    
            if ( len(dists) > 0 ):
                dists.sort()
                dists_nn.append(dists[0])
            else:
                dists_nn.append(np.nan)
                
            if ( len(dists) > 3 ):
                dists_4nn.append(sum(dists[0:4])/4.0)
            else:
                dists_4nn.append(np.nan)
            
        dists_nn = np.array(dists_nn)/pixel_per_mum  
        dists_4nn = np.array(dists_4nn)/pixel_per_mum    
        
        return dists_nn, dists_4nn 
    
    
    
    
    def prepare_data(self):
        
        
        subfolder_name = self.parameters["subfolder_vessel_images"] 
        key_file = self.parameters["meta_data_file"] 
        if (len(self.parameters["macrophage_channel"]) > 1):
            GFP_flag = True
        else:
            GFP_flag = False
        
        overview_file = pd.read_excel(self.rootfolder_name + key_file)
        
        type_column = self.parameters["type_column"]
        
        
        position_col_channel1 = self.parameters["macrophage_position_files"][0]
        if(GFP_flag):
            position_col_channel2 = self.parameters["macrophage_position_files"][1]
        
        #if (GFP_flag == False):
        #    overview_file = overview_file.rename(columns={"analysis_file_macrophages": position_col_channel1, type_column : "type_column"})
        
        subfolder_name = self.rootfolder_name + subfolder_name
        
        print(overview_file.columns)
        
        overview_file  = overview_file.drop([0,1,20])

        overview_file['analysis_file_vessels'] = [subfolder_name + fn_['analysis_file_vessels'] for _, fn_ in
                                                overview_file.iterrows()]
        overview_file[position_col_channel1] = [subfolder_name + fn_[position_col_channel1] for _, fn_ in
                                                    overview_file.iterrows()]
        
        if (GFP_flag):
            overview_file[position_col_channel2] = [subfolder_name + fn_[position_col_channel2] for _, fn_ in
                                                        overview_file.iterrows()]

        df_images = pd.DataFrame()
        df_images.index.name = 'vessel_file'
        df_cells = pd.DataFrame()

        width_threshold_um=10

        for _, fn_ in overview_file.iterrows():
            MP_pos = pd.read_csv(fn_[position_col_channel1])
            vessel_im_all = (plt.imread(fn_['analysis_file_vessels']) > 0).astype(int)
            
            
            print("Size of vessel image: ")
            print(np.array(vessel_im_all).shape)
            pixel_dimension = np.array(vessel_im_all).shape[0]
            
            pixel_per_mum = pixel_dimension/fn_['x_resolution [um]']

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
                vessel_widths_1[vessel_label == l] = np.min(np.array([1. / rp[l - 1]['mean_intensity'], 2*np.sqrt(rp[l-1]['area']/np.pi)]))*(fn_['x_resolution [um]']/pixel_dimension)
                vessel_widths_2[vessel_label == l] = rp[l - 1]['minor_axis_length']*(fn_['x_resolution [um]']/pixel_dimension)

            vessel_im_thin = np.array(vessel_widths_1 > 0) * np.array(vessel_widths_1 <= width_threshold_um) * vessel_widths_1
            vessel_im_thick = np.array(vessel_widths_1 > 0) * np.array(vessel_widths_1 > width_threshold_um) * vessel_widths_1

            # use distance transform to obtain distances of all macrophages from the vessels
            vessel_im_dist_transform = distance_transform_edt(vessel_im_all)*fn_['x_resolution [um]']/pixel_dimension
            vessel_im_dist_thin_transform = distance_transform_edt((vessel_im_thin==0).astype(int))*fn_['x_resolution [um]']/pixel_dimension
            vessel_im_dist_thick_transform = distance_transform_edt((vessel_im_thick==0).astype(int))*fn_['x_resolution [um]']/pixel_dimension

            vessel_number = len(np.unique(vessel_widths_1))-1
            vessel_number_thin = len(np.unique(vessel_im_thin))-1
            vessel_number_thick = len(np.unique(vessel_im_thick))-1

            distances = vessel_im_dist_transform[MP_pos['X'], MP_pos['Y']]
            distances_thin = vessel_im_dist_thin_transform[MP_pos['X'], MP_pos['Y']]
            distances_thick = vessel_im_dist_thick_transform[MP_pos['X'], MP_pos['Y']]
            distances_norm = distances/np.mean(vessel_im_dist_transform)
            
            
            norm_value = float(np.sum(vessel_im_dist_transform))/float(np.count_nonzero(vessel_im_dist_transform))
            
            distances_norm_excl_vsl = distances/norm_value
            
            
           
            occupancy, occupancy_with_vessels = self.calculate_occupancy(MP_pos,vessel_im_dist_transform, self.parameters["occupancy_radius"], pixel_per_mum)
           
            dists_nn, dists_4nn = self.calculate_nearest_neighbours(MP_pos, pixel_per_mum )

            df_cells = df_cells.append(pd.DataFrame({'distance_vessels': distances,
                                            'distance_thin_vessel': distances_thin,
                                            'distance_thick_vessel': distances_thick,
                                            'distance_vessels_norm': distances_norm,
                                            'distance_vessels_norm(excl.vsl.)': distances_norm_excl_vsl,
                                            'occupancy_' : occupancy,
                                            'occupancy_with_vessels' : occupancy_with_vessels,
                                            'nearest_neighbour' : dists_nn,
                                            '4nearest_neighbour' : dists_4nn,
                                            'vessel_file': fn_['analysis_file_vessels'],
                                            'vessel_density': 1 - np.sum(vessel_im_all) / (np.shape(vessel_im_all)[0] * np.shape(vessel_im_all)[1]),
                                            'tumor_type': fn_[type_column],
                                            'MP_type': 'all',
                                            'mouse_name': str(fn_['mouse_name'])}))
            
            if not GFP_flag:
                continue
            
            if (not fn_[position_col_channel2].endswith('None')):
                print(fn_[position_col_channel2])
                MP_pos_GFP = pd.read_csv(fn_[position_col_channel2])
                distances_GFP = vessel_im_dist_transform[MP_pos_GFP['X'], MP_pos_GFP['Y']]
                distances_thin_GFP = vessel_im_dist_thin_transform[MP_pos_GFP['X'], MP_pos_GFP['Y']]
                distances_thick_GFP = vessel_im_dist_thick_transform[MP_pos_GFP['X'], MP_pos_GFP['Y']]

                distances_GFP_norm = distances_GFP/np.mean(vessel_im_dist_transform)
                distances_thin_GFP_norm = distances_thin_GFP/np.mean(vessel_im_dist_thin_transform)
                distances_thick_GFP_norm = distances_thick_GFP/np.mean(vessel_im_dist_thick_transform)
                
                norm_value = float(np.sum(vessel_im_dist_transform))/float(np.count_nonzero(vessel_im_dist_transform))
              
                distances_GFP_norm_excl_vsl = distances_GFP/norm_value               
                occupancy, occupancy_with_vessels = self.calculate_occupancy(MP_pos_GFP,vessel_im_dist_transform, self.parameters["occupancy_radius"], pixel_per_mum)
                
                
                dists_nn, dists_4nn = self.calculate_nearest_neighbours(MP_pos_GFP, pixel_per_mum)

                df_cells = df_cells.append(pd.DataFrame({'distance_vessels': distances_GFP,
                                                'distance_thin_vessel': distances_thin_GFP,
                                                'distance_thick_vessel': distances_thick_GFP,
                                                'distance_vessels_norm': distances_GFP_norm,
                                                'distance_vessels_norm(excl.vsl.)': distances_GFP_norm_excl_vsl,
                                                'occupancy' : occupancy,
                                                'occupancy_with_vessels' : occupancy_with_vessels,
                                                'nearest_neighbour' : dists_nn,
                                                '4nearest_neighbour' : dists_4nn,
                                                'vessel_file': fn_['analysis_file_vessels'],
                                                'vessel_density': 1 - np.sum(vessel_im_all) / (np.shape(vessel_im_all)[0] * np.shape(vessel_im_all)[1]),
                                                'tumor_type': fn_[type_column],
                                                'MP_type': 'implanted',
                                                'mouse_name': str(fn_['mouse_name'])}))

                df_images = df_images.append(pd.DataFrame({'vessel_number': vessel_number,
                            'vessel_number_thin': vessel_number_thin,
                            'vessel_number_thick': vessel_number_thick,
                            'vessel_width_im': [vessel_widths_1],
                            'vessel_width_im_thin': [vessel_im_thin],
                            'vessel_width_im_thick': [vessel_im_thick],
                            'vessel_minor_axis_im': [vessel_widths_2],
                            'tumor_type': fn_[type_column],
                            'distance': np.mean(distances),
                            'distance_thin': np.mean(distances_thin),
                            'distance_thick': np.mean(distances_thick),
                            'distance_GFP': np.mean(distances_GFP),
                            'distance_thin_GFP': np.mean(distances_thin_GFP),
                            'distance_thick_GFP': np.mean(distances_thick_GFP),
                            'distance_norm': np.mean(distances_norm),
                            'distance_GFP_norm': np.mean(distances_GFP_norm),
                            'mouse_name': str(fn_['mouse_name'])}, index=[fn_['analysis_file_vessels']]))
            else:
                df_images = df_images.append(pd.DataFrame({'vessel_number': vessel_number,
                    'vessel_number_thin': vessel_number_thin,
                    'vessel_number_thick': vessel_number_thick,
                    'vessel_width_im': [vessel_widths_1],
                    'vessel_width_im_thin': [vessel_im_thin],
                    'vessel_width_im_thick': [vessel_im_thick],
                    'vessel_minor_axis_im': [vessel_widths_2],
                    'tumor_type': fn_[type_column],
                    'distance': np.mean(distances),
                    'distance_thin': np.mean(distances_thin),
                    'distance_thick': np.mean(distances_thick),
                    'mouse_name': str(fn_['mouse_name'])}, index=[fn_['analysis_file_vessels']]))

        df_cells['mouse_name_MP'] = [str(n_['mouse_name']) + ', ' + n_['MP_type'] for _, n_ in df_cells.iterrows()]

        return df_cells, df_images
