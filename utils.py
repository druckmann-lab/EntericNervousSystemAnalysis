import numpy as np
import spatial_analysis_utils as utils0
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
import os
import matplotlib
import copy
from sklearn.metrics import auc as sklearn_auc
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def quick_save_fig(folder, filename,fig_like,get_face_color=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder,filename)
    if not get_face_color:
        fig_like.savefig(path, format=filename.split('.')[-1],transparent=True)
    else:
        fig_like.savefig(path, format=filename.split('.')[-1],transparent=True, facecolor=fig_like.get_facecolor())
    try:
        plt.close(fig_like)
    except:
        pass

def get_interpolation_mean(hist1d, bin_center_vec_in_microns,interpolation_bin_number=500):
    m = copy.deepcopy(hist1d)
    f = interp1d(copy.deepcopy(bin_center_vec_in_microns), m, kind='cubic')
    bin_center_vec_in_microns_thick = np.linspace(
            np.min(bin_center_vec_in_microns),
            np.max(bin_center_vec_in_microns),
            interpolation_bin_number
            )
    m_thick = f(bin_center_vec_in_microns_thick)
    return m_thick, bin_center_vec_in_microns_thick

def edgeVec2binCenterVec(edgeVec):
    binCenterVec = edgeVec+np.mean(np.diff(edgeVec))/2
    binCenterVec = binCenterVec[:-1]
    return binCenterVec

def edgeVec2midPoint(edgeVec):
    return np.floor((edgeVec.shape[0]-1)/2).astype('int')

def get_conditional_intensity_histogram_rectangular(norm_array, x_size=0.25, y_size=0.08, unit_n_step = 100):
    # norm_arry (n_sample, 2)
    sizes = x_size, y_size
    del x_size
    del y_size

    edges = [np.linspace(-sizes[i],sizes[i],int(2*sizes[i]*unit_n_step+0.5)) for i in [0,1]]


    mid_points = [edgeVec2midPoint(edges[i]) for i in [0,1]]
    

    all_dist_mat = utils0.get_all_to_all_loc_diff_mat(norm_array)
    #print(all_dist_mat.shape)
    H, x_edges, y_edges = np.histogram2d(all_dist_mat[:,0], all_dist_mat[:,1], bins=edges)
    
    # Remove self distance
    H[mid_points[0],mid_points[1]] = H[mid_points[0],mid_points[1]]-norm_array.shape[0]


    return H, x_edges, y_edges

def normH(H):
    uniform_expected_number = np.sum(H[:])/(H.shape[0]*H.shape[1])
    return H/uniform_expected_number

def plot_conditional_intensity_histogram(H, bin_center_vecs, ax=None,fontsize=11):

    xy_range = [bin_center_vecs[i][-1]-bin_center_vecs[i][0] for i in [0,1]]
    
    if ax is None:
        f = plt.figure(figsize=(6,6*xy_range[1]/xy_range[0]))
        ax = plt.subplot(1,1,1)
    
    im = ax.imshow(H,clim=(0.75,3),origin='lower')

    tick_vecs = []

    ii = 0
    tick_vecs.append([0,len(bin_center_vecs[ii])//2,len(bin_center_vecs[ii])-1])
    ax.set_xticks(tick_vecs[ii])
    _ = ax.set_xticklabels(np.round(bin_center_vecs[ii][tick_vecs[ii]]).astype(int))
    ax.set_xlabel('Distance ($\mu$m)',fontsize=fontsize)

    ii = 1
    tick_vecs.append([0,len(bin_center_vecs[ii])//2,len(bin_center_vecs[ii])-1])
    ax.set_yticks(tick_vecs[ii])
    _ = ax.set_yticklabels(np.round(bin_center_vecs[ii][tick_vecs[ii]]).astype(int))
    ax.set_ylabel('Distance ($\mu$m)',fontsize=fontsize)

    return im

def smooth(v,sigma = 1.):
    # Gaussian Kernel Smoothing https://en.wikipedia.org/wiki/Kernel_smoother
    # v: an array or a list
    def __ker(x1,x2,sigma):
        return np.exp(-(x1-x2)**2/(2*sigma**2))
    if sigma == 0:
        return v
    length = len(v)
    xall = np.arange(length)*1.
    smoo_v = np.array([0. for _ in range(length)])
    for x in range(length):
        kernel = __ker(x,xall,sigma)
        den = sum(kernel)
        smoo_v[x] = np.dot(kernel,v)/den
    return smoo_v


def doHist1d(binCenterVecMicronXY,hist,mids,ax=None,doTitle=True, fontsize=11,titleAdd='',doSecPeak=True, doMin=True,interpolation_bin_number=500,smoothSigma=0):


    if ax is not None:
        ax.set_xlim(binCenterVecMicronXY[0][0],binCenterVecMicronXY[0][-1])

    hist1d = histToHist1d(hist, mids)

    hist1dThick, binCenterVecMicronThickX = get_interpolation_mean(hist1d, binCenterVecMicronXY[0],interpolation_bin_number=interpolation_bin_number)

    if smoothSigma > 0:
        hist1dThick = smooth(hist1dThick, smoothSigma)

    bin_center_vec_in_microns = binCenterVecMicronThickX
    # bin_center_vec_in_microns = binCenterVecMicronXY[0]

    (min_point_left,
    second_max_point_left,
    min_point_right,
    second_max_point_right,
    half_max_left_ind,
    half_max_right_ind) = get_key_points(hist1dThick)

    # stripe_stats_dict = utils0.get_stripe_stats_dict(hist1dThick, bin_center_vec_in_microns)

    if (min_point_left is None) or (min_point_right) is None:
        doMin = False
        doSecPeak = False

    title = titleAdd
    width, distance = None, None
    def fetchXY(inds):
        xs = []
        ys = []
        for ind in inds:
            if ind is not None:
                xs = bin_center_vec_in_microns[inds]
                ys = hist1dThick[inds]
            # ind is None iff the min/max point was not found.
        return xs, ys
    if doMin:
        xs, ys = fetchXY([min_point_left,min_point_right])
        if (ax is not None) and len(xs):
            ax.scatter(xs,ys,color='b')
        # ax.scatter(,,color='b')

        xs, ys = fetchXY([half_max_left_ind,half_max_right_ind])

        if len(xs) == 2:
            width = xs[1] - xs[0]
        else:
            width = None
            # it means the at least one of left/right min was not found.

        if (ax is not None) and len(xs):
            ax.scatter(xs,ys,color='k')
            ax.plot(xs,ys,color='k')
            if width is not None:
                title += "Stripe Width: {:.1f} $\mu$m".format(width)
            else:
                title += "Stripe Width not Available"
    if doSecPeak:
        xs, ys = fetchXY([second_max_point_left,second_max_point_right])

        if len(xs) == 2:
            distance = (xs[1] - xs[0])/2
        else:
            distance = None
            # it means the at least one of left/right sec max was not found.
        if ax is not None:
            ax.scatter(xs,ys,color='r')
            ax.plot(xs,ys,color='r')
            if distance is not None:
                title += "Stripe Distance: {:.1f} $\mu$m".format(distance)
            else:
                title += "Stripe Distance not Available"



    if ax is not None:
        ax.plot(binCenterVecMicronThickX,hist1dThick)
        ax.set_xlabel('$\mu$m', fontsize=fontsize)
        ax.set_ylabel('prob. density', fontsize=fontsize)
        if doTitle:
            ax.set_title(title, fontsize=fontsize)
    return width, distance


def get_key_points(m_raw):
    m = copy.deepcopy(m_raw)
    m -= m.min()
    m /= m.max()
    hist_size = m.shape[0]
    mid_point = int((hist_size-1)/2)
    min_point_left,second_max_point_left = find_secondary_stripe_one_side(m,np.arange(mid_point,1,-1))
    min_point_right,second_max_point_right = find_secondary_stripe_one_side(m,np.arange(mid_point,hist_size,1))

    if (min_point_left is None) or (min_point_right) is None:
        return [None]*6
    
    half_max_amp = ((m[mid_point]-m[min_point_left])+(m[mid_point]-m[min_point_right]))/4
    half_max_val = (m[min_point_left]+m[min_point_right]+2*half_max_amp)/2
    half_max_left_ind = np.argmin((
        m[np.arange(min_point_left, mid_point)]-half_max_val
        )**2)
    half_max_left_ind += min_point_left

    half_max_right_ind = np.argmin((
        m[np.arange(mid_point, min_point_right)]-half_max_val
        )**2)
    half_max_right_ind += mid_point
    
    return (min_point_left,
        second_max_point_left,
        min_point_right,
        second_max_point_right,
        half_max_left_ind,
        half_max_right_ind)

def find_secondary_stripe_one_side(m, ind_vec):
    min_point = None
    second_max_point= None
    found_min_flag = False
    found_second_max_flag = False
    thre_for_min = 1.0
    for ii in range(3,len(ind_vec)):
        #print(ind_vec[ii])
        if (m[ind_vec[ii]]>m[ind_vec[ii-1]]) & (m[ind_vec[ii]]>m[ind_vec[ii-2]]) & (found_min_flag==False):
            if m[ind_vec[ii-1]] > thre_for_min:
                continue
            min_point = ind_vec[ii-1]
            found_min_flag = True
            #print('Min found at {}'.format(min_point))
    if found_min_flag == False:
        # print('No min found')
        return None, None

    reduced_ind_vec = ind_vec[(np.where(ind_vec==min_point)[0][0]+1):]
    wing = m[min_point:] if min_point > len(m)/2 else m[:min_point]
    mi, mx = m[min_point], np.max(wing)
    thre_for_sec_max = mi + (mx-mi)*0.3
    for ii in range(3,len(reduced_ind_vec)):
        #print(reduced_ind_vec[ii])
        if (m[reduced_ind_vec[ii]]<m[reduced_ind_vec[ii-1]]) & \
        (m[reduced_ind_vec[ii]]<m[reduced_ind_vec[ii-2]]) & (found_second_max_flag==False):
            if m[reduced_ind_vec[ii]] < thre_for_sec_max:
                continue
            second_max_point = reduced_ind_vec[ii-1]
            found_second_max_flag=True
            # print(m[reduced_ind_vec[ii]], thre_for_sec_max)
            #print('Second max found at {}'.format(second_max_point))
    # if found_second_max_flag == False:
    #     print('No sec max found')
    if not found_min_flag:
        min_point = None
    if not found_second_max_flag:
        second_max_point = None
    return min_point, second_max_point

def histToHist1d(hist_raw, mids):
    hist = copy.deepcopy(hist_raw)
    # hist = np.concatenate([hist[:,:mids[1]],hist[:,mids[1]+1:]],axis=1)
    hist[mids[0],mids[1]] = np.mean([hist[mids[0]+1,mids[1]],hist[mids[0],mids[1]+1],hist[mids[0]-1,mids[1]],hist[mids[0],mids[1]-1]])
    hist1d = np.mean(hist,axis=1)

    
    # don't count the self point
    # print(mids[0],hist1d.shape)
    # hist1d[mids[0]] = np.mean([hist1d[mids[0]-1],hist1d[mids[0]+1]])
    return hist1d


def get_Euclidean_distance_btw_nd_points(a, b):
    '''
    get Euclidiean distance between n-dimensional points
    a: (num_points_a, num_dim) e.g. (10, 2)
    b: (num_points_b, num_dim) e.g. (20, 2)
    return dist_mat: (10, 20)
    '''
    assert a.shape[-1] == b.shape[-1]
    n_dim = a.shape[-1]
    return np.sqrt(np.sum(
        [(
            a[:,d] - b[:,d].reshape(-1,1)
            )**2 for d in range(n_dim)],
        axis=0))

def denorm_arr(arr,x_span, y_span):
    arr2 = np.zeros_like(arr)
    arr2[:,0] = arr[:,0]*x_span
    arr2[:,1] = arr[:,1]*y_span
    return arr2


def get_Euclidean_distance_btw_2d_points_with_span(a,b,x_span, y_span):
    a = denorm_arr(a, x_span, y_span)
    b = denorm_arr(b, x_span, y_span)
    dist_mat = get_Euclidean_distance_btw_nd_points(a,b)
    return dist_mat

def mask_tf_mat(tf_mat):
    mask = np.ones_like(tf_mat)
    # 1 1 1
    # 1 1 1
    # 1 1 1
    mask = np.tril(mask)
    # 1 0 0
    # 1 1 0
    # 1 1 1
    mask = mask == 0
    # F T T
    # F F T
    # F F F
    tf_mat = tf_mat * mask
    return tf_mat

def get_good_samples_from_dist_mat(dist_mat, minimal_distance ,is_self_collision):
    tf_mat = dist_mat < minimal_distance
    if is_self_collision:
        tf_mat = mask_tf_mat(tf_mat)
    overlapped_samples = np.any(tf_mat, axis=1)
    good_samples = ~overlapped_samples
    return good_samples

def filter_out_collision(rand_arr1, rand_arr2, minimal_distance, x_span, y_span, is_self_collision):
    dist_mat = get_Euclidean_distance_btw_2d_points_with_span(rand_arr1, rand_arr2, x_span, y_span)
    good_samples = get_good_samples_from_dist_mat(dist_mat, minimal_distance, is_self_collision)
    return rand_arr2[good_samples]

def draw_random_locations_with_minimal_distance(density_per_unit_area,x_span,y_span, minimal_distance):
    

    # print(x_span, y_span)
    square_face_length=1 # To simplfy the problem, square_face_length must be 1.

    current_density = density_per_unit_area
    random_loc_array = np.zeros((0,2))

    while 1:
        new_random_loc_array = utils0.draw_random_locations(current_density, square_face_length)
        if minimal_distance == 0:
            return new_random_loc_array
        generated_len= len(new_random_loc_array)

        # check self collision
        if generated_len > 1:
            new_random_loc_array = filter_out_collision(new_random_loc_array, new_random_loc_array, minimal_distance, x_span, y_span, is_self_collision=True)


        # check old-new collision
        if (random_loc_array.shape[0] > 0) and (generated_len > 0):
            new_random_loc_array = filter_out_collision(random_loc_array, new_random_loc_array, minimal_distance, x_span, y_span, is_self_collision=False)

        random_loc_array = np.concatenate([random_loc_array, new_random_loc_array])

        
        cut_len = generated_len - len(new_random_loc_array)
        if cut_len == 0:
            # if no collision happen during this epoch then it's finished.
            return random_loc_array
        else:
            
            current_density = current_density * cut_len / generated_len

def get_cumulative_nearest_nbr_min_distance(dist_x_vec, point_density, sample_num, x_span, y_span, minimal_distance, area, perimeter):
    h_hat_array = np.zeros((dist_x_vec.shape[0],sample_num))
    z_random_more = []
    #print(h_hat_array.shape)
    for ii in range(0,sample_num):
        # random_loc_array = draw_random_locations(point_density, 1) # Uses normalized area
        random_loc_array = draw_random_locations_with_minimal_distance(point_density, x_span, y_span, minimal_distance)
        min_dist_array, min_dist_ind = utils0.find_nearest_neighbor_and_distance(random_loc_array)
        h_emp_random_data = utils0.empirical_cumulative_count_by_distance(dist_x_vec, min_dist_array)
        h_hat_array[:,ii] = np.copy(h_emp_random_data)

        m_random, v_random = utils0.get_statistics_for_mean_nearest_neighbor_distance(\
                        min_dist_array.shape[0], area, perimeter)
        z_random_more.append((np.mean(min_dist_array)-m_random)/np.sqrt(v_random))
    return h_hat_array, np.array(z_random_more)

def data_spatial_randomness_comparison_summary(
    data_loc_array, 
    bin_point_num,
    sample_num_for_random_comparison,
    minimal_distance=0,
    do_plot=True,
    fontsize=15
    ):
    area = 1 # Normalized areas
    perimeter = 4 # Normalized areas
    
    norm_loc_array,x_span,y_span = utils0.simple_normalize_data(data_loc_array)
    dist_bin_vec = np.linspace(0,0.1,bin_point_num)
    dist_x_vec = dist_bin_vec + np.mean(np.diff(dist_bin_vec))/2
    
    point_density = norm_loc_array.shape[0]
    # random_loc_array = utils0.draw_random_locations(point_density, 1)

    if isinstance(minimal_distance,str):
        assert minimal_distance == 'auto_adjust'
        dist_mat = get_Euclidean_distance_btw_2d_points_with_span(norm_loc_array,norm_loc_array,x_span,y_span)
        np.fill_diagonal(dist_mat,np.inf)
        minimal_distance = np.min(dist_mat)
    random_loc_array = draw_random_locations_with_minimal_distance(point_density, x_span, y_span, minimal_distance)
    # random_loc_array = utils0.draw_random_locations(point_density, 1)


    # minimal_distance = 
    
    min_dist_array_data, _ = utils0.find_nearest_neighbor_and_distance(norm_loc_array)
    min_dist_array_random, _ = utils0.find_nearest_neighbor_and_distance(random_loc_array)
    
    m_data, v_data = utils0.get_statistics_for_mean_nearest_neighbor_distance(\
                        min_dist_array_data.shape[0], area, perimeter)
    m_random, v_random = utils0.get_statistics_for_mean_nearest_neighbor_distance(\
                        min_dist_array_random.shape[0], area, perimeter)
    # print('Data:')
    z_data = (np.mean(min_dist_array_data)-m_data)/np.sqrt(v_data)
    # print((np.mean(min_dist_array_data)-m_data)/np.sqrt(v_data))
    z_random = (np.mean(min_dist_array_random)-m_random)/np.sqrt(v_random)
    # print((np.mean(min_dist_array_random)-m_random)/np.sqrt(v_random))


    histcount_data = np.histogram(min_dist_array_data,dist_bin_vec)
    histcount_random = np.histogram(min_dist_array_random,dist_bin_vec)
    
    h_emp_data = utils0.empirical_cumulative_count_by_distance(dist_x_vec, min_dist_array_data)
    h_hat_nearest_neighbor_random, z_random_more = get_cumulative_nearest_nbr_min_distance(dist_x_vec, point_density, sample_num_for_random_comparison, x_span, y_span, minimal_distance, area, perimeter)
    theoretical_nn_cdf = \
        np.mean(h_hat_nearest_neighbor_random,axis=1) # Can use approximation based on 1-exp, they look similar, see below

    n_rand = len(h_hat_nearest_neighbor_random)
    # theoretical_nn_cdf (50,)
    # h_hat_nearest_neighbor_random (50, 100)
    auc_rand = np.array([sklearn_auc(theoretical_nn_cdf,x) for x in h_hat_nearest_neighbor_random.swapaxes(0,1)])
    auc_data = sklearn_auc(theoretical_nn_cdf,h_emp_data)
    a_perc = np.mean((auc_rand - auc_data)>0)

    z_perc =  np.mean((z_data - z_random_more) > 0)

    if not do_plot:
        return None, None, z_data, z_random, a_perc, z_perc, z_random_more


    f = plt.figure(figsize=(12,12))
    subplot_num = 4
    axis_list = list()
    for ii in range(subplot_num):
        a = plt.subplot(2,2,ii+1)
        axis_list.append(a)

    axis_list[0].scatter(norm_loc_array[:,0], norm_loc_array[:,1],c='b',s=10)
    # axis_list[0].set_title('Data', fontsize=fontsize)
    axis_list[0].set_xlabel('x (in norm. units)', fontsize=fontsize)
    axis_list[0].set_ylabel('y (in norm. units)', fontsize=fontsize)

    axis_list[1].scatter(random_loc_array[:,0], random_loc_array[:,1],c='k',s=10)
    axis_list[1].set_title('Random sample with minimal distance {:.2f}'.format(minimal_distance), fontsize=fontsize)
    axis_list[1].set_xlabel('x (in norm. units)', fontsize=fontsize)
    axis_list[1].set_ylabel('y (in norm. units)', fontsize=fontsize)

    axis_list[2].step(histcount_data[1], np.append(0,histcount_data[0]),'b',label='data')
    axis_list[2].step(histcount_random[1], np.append(0,histcount_random[0]),'k',label='random')
    #plt.plot([np.mean(min_dist_array_data), np.mean(min_dist_array_data)],axis_list[2].get_ylim)
    axis_list[2].set_xlabel('distance (norm.)', fontsize=fontsize)
    axis_list[2].set_ylabel('Count', fontsize=fontsize)
    axis_list[2].set_title('Nearest neighbor histogram', fontsize=fontsize)
    leg = axis_list[2].legend(loc='upper right', frameon=False)
  
    axis_list[3].plot(theoretical_nn_cdf,h_emp_data,'b',label='data')
    axis_list[3].plot((0,1),(0,1),'--k')
    axis_list[3].plot(theoretical_nn_cdf,np.max(h_hat_nearest_neighbor_random,axis=1),color='gray',label='random max')
    axis_list[3].plot(theoretical_nn_cdf,np.min(h_hat_nearest_neighbor_random,axis=1),color='gray',label='random min')
    axis_list[3].set_xlabel('Theoretical CDF value', fontsize=fontsize)
    axis_list[3].set_ylabel('Empirical CDF value', fontsize=fontsize)



    axis_list[3].set_title('z_data {:.2f} z_rand {:.2f} z_perc{:.1f}% a_perc {:.1f}%'.format(z_data, z_random, z_perc, a_perc*100), fontsize=fontsize)
    leg = axis_list[3].legend(loc='upper left', frameon=False, fontsize=fontsize)
    
    for ax in axis_list:
        ax.tick_params(labelsize=fontsize)

    
    f.tight_layout()
    
    return axis_list, f, z_data, z_random, a_perc, z_perc, z_random_more