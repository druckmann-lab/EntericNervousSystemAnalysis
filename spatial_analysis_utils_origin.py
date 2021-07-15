import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp1d
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def load_multi_sheet_data(data_file):
    # load
    xls = pd.ExcelFile(data_file)
    loc_array_list = list()
    sheet_num = len(xls.sheet_names)
    for ii in range(0,sheet_num):
        print(ii)
        data = pd.read_excel(data_file, sheet_name=xls.sheet_names[ii])
        loc_array = data.loc[:,['X','Y']].to_numpy()
        loc_array_list.append(np.copy(loc_array))
    return loc_array_list

def simple_normalize_data(loc_array):
    # normalize
    norm_loc_array = np.copy(loc_array.astype(np.float64))
    x_max, y_max = loc_array.max(axis=0)
    x_min, y_min = loc_array.min(axis=0)
    x_span = x_max-x_min
    y_span = y_max-y_min

    norm_loc_array[:,0] = norm_loc_array[:,0]-x_min
    norm_loc_array[:,0] = norm_loc_array[:,0]/(x_span)
    norm_loc_array[:,1] = norm_loc_array[:,1]-y_min
    norm_loc_array[:,1] = norm_loc_array[:,1]/(y_span)

    return norm_loc_array

def load_and_pre_process_data(data_file, analysis_region_dict):
    # load
    data = pd.read_csv(data_file)
    loc_array = data.loc[:,['X','Y']].to_numpy()
    # crop
    mask = (loc_array[:,0]>analysis_region_dict['x_min'])
    mask = mask*(loc_array[:,0]<analysis_region_dict['x_max'])
    mask = mask*(loc_array[:,1]>analysis_region_dict['y_min'])
    mask = mask*(loc_array[:,1]<analysis_region_dict['y_max'])
    loc_array = loc_array[mask,:]

    # normalize
    loc_array = loc_array.astype(np.float64)
    loc_array[:,0] = loc_array[:,0]-analysis_region_dict['x_min']
    loc_array[:,0] = loc_array[:,0]/(analysis_region_dict['x_max']-analysis_region_dict['x_min'])
    loc_array[:,1] = loc_array[:,1]-analysis_region_dict['y_min']
    loc_array[:,1] = loc_array[:,1]/(analysis_region_dict['y_max']-analysis_region_dict['y_min'])

    return loc_array

def plot_conditional_intensity_and_control(H, H_random, xedges, yedges, clim):
    # Annoying inverse convention for x and y
    fig = plt.figure(figsize=(10,5))
    ax_list = []
    ax_list.append(plt.subplot(1,2,1))
    plt.imshow(H,clim=[0,clim])
    plt.colorbar()

    ax_list.append(plt.subplot(1,2,2))
    plt.imshow(H_random,clim=[0,clim])
    plt.colorbar()

    ax_list[0].set_title('Data')
    ax_list[1].set_title('Random')


    for ii in range(0,2):
        #print(ii)
        x_tick_loc = np.arange(2,20,5)
        ax_list[ii].set_xticks(x_tick_loc)
        x_tick_vals = yedges[x_tick_loc]
        x_tick_vals = np.round(x_tick_vals*100)/100
        ax_list[ii].set_xticklabels(x_tick_vals)

        y_tick_loc = np.arange(2,20,5)
        ax_list[ii].set_yticks(y_tick_loc)
        y_tick_vals = xedges[y_tick_loc]
        y_tick_vals = np.round(y_tick_vals*100)/100
        ax_list[ii].set_yticklabels(y_tick_vals)
    return

def plot_data_and_random_scatter_plots(norm_loc_array, random_loc_array):
    plt.figure(figsize=(12,6))
    subplot_num = 2
    axis_list = list()
    for ii in range(subplot_num):
        a = plt.subplot(1,2,ii+1)
        axis_list.append(a)

    axis_list[0].scatter(norm_loc_array[:,0], norm_loc_array[:,1],c='b',s=10)
    axis_list[0].set_title('Data')

    axis_list[1].scatter(random_loc_array[:,0], random_loc_array[:,1],c='k',s=10)
    axis_list[1].set_title('Random sample')
    return

def get_all_to_all_loc_diff_matrix(norm_loc_array):
    neuron_num = norm_loc_array.shape[0]
    all_to_all_loc_diff_mat = np.zeros((neuron_num*neuron_num,2))
    for ii in range(0,neuron_num):
        all_to_all_loc_diff_mat[(ii*neuron_num):((ii+1)*neuron_num)] = \
            np.copy(norm_loc_array)-norm_loc_array[ii,:]
    tiny = 1e-10
    all_to_all_loc_diff_mat = all_to_all_loc_diff_mat[np.sum(np.abs(all_to_all_loc_diff_mat),axis=1)>tiny,:]
    return all_to_all_loc_diff_mat

def find_nearest_neighbor_and_distance(loc_array):
    point_num = loc_array.shape[0]
    min_dist_array = np.zeros(point_num)
    min_dist_ind = np.zeros(point_num)
    for ii in range(0,point_num):
        m = np.tile(loc_array[ii,:],(loc_array.shape[0],1))
        d = loc_array-m
        d = np.sqrt(np.sum(d**2,1))
        d[ii]=1e5
        min_pos = d.argmin()
        min_dist_ind[ii] = min_pos
        min_dist_array[ii] = d[min_pos]
    return min_dist_array, min_dist_ind

def draw_random_locations(density_per_unit_area, square_face_length):
    random_point_num = np.random.poisson(density_per_unit_area*(square_face_length**2))
    random_loc_array = np.random.uniform(0,square_face_length,random_point_num*2)
    random_loc_array = np.reshape(random_loc_array,(random_point_num,2))
    return random_loc_array

def expected_distribution_function_on_square(bin_vec):
    # bin_vec expected to be between 0 and sqrt(2) that is max of dist in square of face length 1
    edf = np.zeros(bin_vec.shape[0])
    ind_one = np.where(bin_vec<=1)[0]
    edf_one = np.pi*(bin_vec[ind_one]**2)-8*(bin_vec[ind_one]**3)/3+(bin_vec[ind_one]**4)/2
    edf[ind_one] = edf_one
    #ind_two = np.where((bin_vec>1) and (bin_vec<=np.sqrt(2)))[0]
    ind_two = np.arange(0,bin_vec.shape[0])[(bin_vec>1)*(bin_vec<np.sqrt(2))]
    edf_two = (1/3)-2*(bin_vec[ind_two]**2) - (bin_vec[ind_two]**4)/2 + \
        (4/3)*np.sqrt((bin_vec[ind_two]**2)-1)*(2*(bin_vec[ind_two]**2)+1) + \
        2*(bin_vec[ind_two]**2)*np.arcsin(2*(bin_vec[ind_two]**-2)-1)
    edf[ind_two] = edf_two
    return edf

def empirical_cumulative_count_by_distance(dist_x_vec, min_dist_vec):
    # lazy way
    emp_cum_val = np.zeros(dist_x_vec.shape[0])
    for ii in range(0,dist_x_vec.shape[0]):
        emp_cum_val[ii] = np.sum(min_dist_vec<dist_x_vec[ii])
    emp_cum_val = emp_cum_val/min_dist_vec.shape[0]
    return emp_cum_val

def get_conditional_intensity_histogram(norm_loc_array, edge_vec):
    x_edges = edge_vec
    y_edges = edge_vec

    mid_point = np.floor((edge_vec.shape[0]-1)/2).astype('int')

    all_dist_mat = get_all_to_all_loc_diff_matrix(norm_loc_array)
    #print(all_dist_mat.shape)
    H, xedges, yedges = np.histogram2d(all_dist_mat[:,0], all_dist_mat[:,1], bins=(x_edges, y_edges))
    
    # Remove self distance
    H[mid_point,mid_point] = H[mid_point,mid_point]-norm_loc_array.shape[0]
    # The commented out naïve thing below is misleading since it assumes all points enter
    # whereas bin vecs limit that
    #uniform_expected_number = all_dist_mat.shape[0]/((len(x_edges)-1)**2)
    uniform_expected_number = np.sum(H[:])/((len(x_edges)-1)**2)
    #print(uniform_expected_number)
    H = H/uniform_expected_number
    H = H.T
    
    return H


def get_large_empirical_cumulative_count(dist_x_vec, point_density, sample_num):
    h_hat_array = np.zeros((dist_x_vec.shape[0],sample_num))
    print(h_hat_array.shape)
    for ii in range(0,sample_num):
        random_loc_array = draw_random_locations(point_density, 1) # Uses normalized area
        d_r = sp.spatial.distance.cdist(random_loc_array, random_loc_array)
        d_r = d_r.reshape(d_r.shape[0]**2)
        d_r = d_r[d_r!=0]
        h_emp_random_data = empirical_cumulative_count_by_distance(dist_x_vec, d_r)
        h_hat_array[:,ii] = np.copy(h_emp_random_data)
    return h_hat_array

def get_large_empirical_cumulative_count_nearest_neighbor(dist_x_vec, point_density, sample_num):
    h_hat_array = np.zeros((dist_x_vec.shape[0],sample_num))
    #print(h_hat_array.shape)
    for ii in range(0,sample_num):
        random_loc_array = draw_random_locations(point_density, 1) # Uses normalized area
        min_dist_array, min_dist_ind = find_nearest_neighbor_and_distance(random_loc_array)
        h_emp_random_data = empirical_cumulative_count_by_distance(dist_x_vec, min_dist_array)
        h_hat_array[:,ii] = np.copy(h_emp_random_data)
    return h_hat_array

def get_statistics_for_mean_nearest_neighbor_distance(sample_num, area, perimeter):
    mean_for_test  = (1/2)*np.sqrt(((1/sample_num)*area)) + (0.051+0.042*(1/np.sqrt(sample_num)))*(perimeter/sample_num)
    var_for_test = 0.07*(1/(np.power(sample_num,2)))*area + 0.037*( np.sqrt((1/np.power(sample_num,5))*area) )*perimeter
    return mean_for_test, var_for_test

def get_averaged_conditional_intensity_histogram(data_array_list, use_index_list, edge_vec):
    norm_loc_array = simple_normalize_data(data_array_list[use_index_list[0]])
    H = get_conditional_intensity_histogram(norm_loc_array, edge_vec)
    avg_H = np.copy(H)
    for ii in range(1, len(use_index_list)):
        norm_loc_array = simple_normalize_data(data_array_list[use_index_list[ii]])
        H = get_conditional_intensity_histogram(norm_loc_array, edge_vec)
        avg_H = avg_H + np.copy(H)
    avg_H = avg_H/len(use_index_list)
    return avg_H

def plot_conditional_intensity_histogram(H, bin_center_vec, ax):
    if ax==False:
        f = plt.figure(figsize=(6,6))
        ax = plt.subplot(1,1,1)
    
    p = ax.imshow(H,clim=(0.75,3))
    tick_vec = generate_tick_list_for_CIF(bin_center_vec)
    ax.set_xticks(tick_vec)
    _ = ax.set_xticklabels(np.round(bin_center_vec[tick_vec]).astype(int))
    ax.set_xlabel('Distance (um)')
    ax.set_yticks(tick_vec)
    _ = ax.set_yticklabels(np.round(bin_center_vec[tick_vec]).astype(int))
    ax.set_ylabel('Distance (um)')

    #plt.colorbar(p)
    #ax.set_title(region_type_list[10][:-1])
    #plt.savefig('CI_PC.pdf')

def generate_tick_list_for_CIF(bin_center_vec):
    mid_point = np.floor((bin_center_vec.shape[0]-1)/2).astype('int')
    tick_vec = np.append(np.linspace(0,mid_point,3), np.linspace(mid_point,len(bin_center_vec)-1,3))
    tick_vec = np.unique(np.floor(tick_vec).astype(int))
    return tick_vec

def stripe_analysis(CIF_hist, bin_center_vec_in_microns, plot_flag, interpolation_bin_number, save_filename):
    hist = np.copy(CIF_hist)
    hist_size = hist.shape[0]
    mid_point = int((hist_size-1)/2)
    width = 1
    # Hacky way of restoring the missing zero point
    hist[mid_point,mid_point] = (hist[mid_point-1,mid_point]+hist[mid_point+1,mid_point])/2
    
    m = np.mean(hist,axis=0)
    raw_mean = np.copy(m)
    raw_bin_center_vec_in_microns = np.copy(bin_center_vec_in_microns)

    if interpolation_bin_number != None:
        f = interp1d(bin_center_vec_in_microns, m, kind='cubic')
        bin_center_vec_in_microns = np.linspace(\
            np.min(raw_bin_center_vec_in_microns), np.max(raw_bin_center_vec_in_microns),\
            interpolation_bin_number)
        m = f(bin_center_vec_in_microns)

    min_point_left,second_max_point_left, min_point_right,second_max_point_right, \
    half_max_left_ind, half_max_right_ind = get_key_points(m)
    
    stripe_stats_dict = get_stripe_stats_dict(m, bin_center_vec_in_microns)
    if plot_flag:
        
        f = plt.figure(figsize = (6,9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax_top = plt.subplot(gs[0])
        plot_conditional_intensity_histogram(CIF_hist, raw_bin_center_vec_in_microns, ax_top)
        #ax_top.imshow(CIF_hist, clim=(0.75,3))
        ax = plt.subplot(gs[1])
        plt.tight_layout()

        ax.plot(bin_center_vec_in_microns, m)
        ax.scatter(bin_center_vec_in_microns[second_max_point_left],m[second_max_point_left],color='r')
        ax.scatter(bin_center_vec_in_microns[second_max_point_right],m[second_max_point_right],color='r')

        ax.scatter(bin_center_vec_in_microns[min_point_left],m[min_point_left],color='b')
        ax.scatter(bin_center_vec_in_microns[min_point_right],m[min_point_right],color='b')

        ax.scatter(bin_center_vec_in_microns[half_max_left_ind],m[half_max_left_ind],color='k')
        ax.scatter(bin_center_vec_in_microns[half_max_right_ind],m[half_max_right_ind],color='k')
        ax.plot((bin_center_vec_in_microns[half_max_left_ind], bin_center_vec_in_microns[half_max_right_ind]), \
                (m[half_max_left_ind], m[half_max_right_ind]),color='k')
        if save_filename:
            plt.savefig(save_filename, type='pdf')
    return stripe_stats_dict

def get_stripe_stats_dict(m, bin_center_vec_in_microns):
    stripe_stats_dict = dict()
    hist_size = m.shape[0]
    mid_point = int((hist_size-1)/2)
    
    ind_dist_in_microns = np.mean(np.diff(bin_center_vec_in_microns))
    min_point_left,second_max_point_left, min_point_right,second_max_point_right, \
        half_max_left_ind, half_max_right_ind = get_key_points(m)
    
    stripe_stats_dict['max_to_left_stripe_trough_dist'] = (mid_point-min_point_left)*ind_dist_in_microns
    stripe_stats_dict['max_to_left_stripe_max_dist'] = (mid_point-second_max_point_left)*ind_dist_in_microns

    stripe_stats_dict['max_to_right_stripe_trough_dist'] = (min_point_right-mid_point)*ind_dist_in_microns
    stripe_stats_dict['max_to_right_stripe_max_dist'] = (second_max_point_right-mid_point)*ind_dist_in_microns

    stripe_stats_dict['primary_stripe_width_at_half_max'] =  (half_max_right_ind-half_max_left_ind)*ind_dist_in_microns
    return stripe_stats_dict

def get_key_points(m):
    hist_size = m.shape[0]
    mid_point = int((hist_size-1)/2)
    min_point_left,second_max_point_left = find_secondary_stripe_one_side(m,np.arange(mid_point,1,-1))
    min_point_right,second_max_point_right = find_secondary_stripe_one_side(m,np.arange(mid_point,hist_size,1))
    
    half_max_amp = ((m[mid_point]-m[min_point_left])+(m[mid_point]-m[min_point_right]))/4
    half_max_val = (m[min_point_left]+m[min_point_right]+2*half_max_amp)/2
    half_max_left_ind = np.argmin((m[np.arange(mid_point)]-half_max_val)**2)
    half_max_right_ind = np.argmin((m[np.arange(mid_point,hist_size)]-half_max_val)**2)
    half_max_right_ind = half_max_right_ind+mid_point
    
    return min_point_left,second_max_point_left, min_point_right,second_max_point_right, half_max_left_ind, half_max_right_ind

def find_secondary_stripe_one_side(m, ind_vec):
    min_point = np.nan
    second_max_point=np.nan
    past_min_flag = False
    found_second_max_flag = False
    for ii in range(2,len(ind_vec)):
        #print(ind_vec[ii])
        if (m[ind_vec[ii]]>m[ind_vec[ii-1]]) & (m[ind_vec[ii]]>m[ind_vec[ii-2]]) & (past_min_flag==False):
            min_point = ind_vec[ii-1]
            past_min_flag = True
            #print('Min found at {}'.format(min_point))
    if past_min_flag == False:
        print('No min found')
        #break
    reduced_ind_vec = ind_vec[(np.where(ind_vec==min_point)[0][0]+1):]
    #print(reduced_ind_vec)
    for ii in range(2,len(reduced_ind_vec)):
        #print(reduced_ind_vec[ii])
        if (m[reduced_ind_vec[ii]]<m[reduced_ind_vec[ii-1]]) & \
        (m[reduced_ind_vec[ii]]<m[reduced_ind_vec[ii-2]]) & (found_second_max_flag==False):
            second_max_point = reduced_ind_vec[ii-1]
            found_second_max_flag=True
            #print('Second max found at {}'.format(second_max_point))
    return min_point, second_max_point