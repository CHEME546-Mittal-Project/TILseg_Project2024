# Internal Imports
import os

# External Imports
import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from skimage import color
from skimage.measure import regionprops, label
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.cluster import DBSCAN
import tifffile
from tqdm import tqdm


def nearest_neighbor_distance(binary_mask: np.ndarray, 
                              save_path: str = None, 
                              image_path: str = None,
                              plot_show: bool = False):
    #TODO should I do image path or image name?? how to incorporate csv export and stuff
    """
    For each inidividual TIL, the nearest neighbor distance will be calculated from 
    the binary mask. The results are visualized in a histogram of these distances
    with the frequency indicating the number of TILs that have a nearest neighbor 
    corresponding to that distance.
    
    Parameters
    -----
    binary_mask (np.ndarray): a 2D numpy array with 0's being background and groups
    of 1's indicating TILs. 
    
    save_path (str): path of the folder in which the histogram will be saved. 
    Default is none.
    
    image_path (str): path to the image file (.tif). Default is None
    
    plot_show (bool): If true, the plot will be shown when the function is called.
    If false, it will suppress the display of the plot. Default is False.
    
    Returns
    -----
    mean_distance (float): The mean distance among all TILs in the image.
    
    max_distance (float): The maximum distance among all TILs in the image.
    
    stdev_distance (float): The standard deviation of all distances
    between TILs in the image.
    """
    
    # Label connected components in the binary mask
    labeled_mask = label(binary_mask)
    
    # Compute properties for each labeled region
    regions = regionprops(labeled_mask)
    
    # Extract centroids of each labeled region
    til_centers = [region.centroid for region in regions]
    
    # Calculate pairwise distances between all TIL centers
    distances = distance.cdist(til_centers, til_centers)
    
    # Replace diagonal distances (self-distance) with infinity
    # to excluded these distances
    np.fill_diagonal(distances, np.inf)
    
    # Find the minimum distance for each cluster
    nearest_distances = np.min(distances, axis=1)

    # Calculate mean, max, and std dev of distances
    mean_distance = np.mean(nearest_distances)
    max_distance = np.max(nearest_distances)
    stdev_distance = np.std(nearest_distances)
    
    # Plot histogram of nearest neighbor distances + important labels
    plt.figure(figsize=(10, 6))
    plt.hist(nearest_distances, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=mean_distance, color='red', linestyle='--', label=f'Mean Distance: {mean_distance:.2f}')
    plt.text(mean_distance + 2, plt.ylim()[1] * 0.95, f'Mean: {mean_distance:.2f}', color='black', fontsize=12, ha='left')
    plt.axvline(x=mean_distance + stdev_distance, color='green', linestyle='--', label=f'Standard Deviation: {stdev_distance:.2f}')
    plt.axvline(x=mean_distance + 2 * stdev_distance, color='green', linestyle='--')
    plt.axvline(x=mean_distance - stdev_distance, color='green', linestyle='--')
    plt.xlabel('Nearest Neighbor Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Nearest Neighbor Distances for TIL Clusters')
    
    if save_path:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{image_name}_nn_histogram.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
    
    if plot_show:
        plt.show()
    else:
        plt.close()

    return mean_distance, max_distance, stdev_distance


def til_size(binary_mask, 
             save_path: str = None, 
             image_path: str = None,
             show_plot: bool = False): 
    #TODO should I do image path or image name?? how to incorporate csv export and stuff
    """
    For each inidividual TIL, the area size (in pixels) will be calculated from the binary 
    mask. The results are visualized in a histogram of these sizes with the frequency 
    indicating the number of TILs that correspond to that size.
    
    Parameters
    -----
    binary_mask (np.ndarray): a 2D numpy array with 0's being background and groups
    of 1's indicating TILs. 
    
    save_path (str): path of the folder in which the histogram will be saved. 
    Default is none.
    
    image_path (str): path to the image file (.tif). Default is None
   
    plot_show (bool): If true, the plot will be shown when the function is called.
    If false, it will suppress the display of the plot. Default is False.

    Returns
    -----
    mean_size (int): The mean pixel size of all TILs in the image.
   
    min_til_size (int): The smallest TIL size of all TILs in the image.
    
    til_count (int): The total number of TILs in the image.
    """
    
    # Convert the binary mask to uint8
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

    # Find connected components in the binary mask
    til_count, labels = cv2.connectedComponents(binary_mask_uint8)

    # Calculate the size of each TIL 
    unique_labels, til_sizes = np.unique(labels, return_counts=True)
    til_sizes = til_sizes[1:]  # Exclude background (label 0)
    mean_size = np.mean(til_sizes)

    # Plot a histogram of TIL sizes
    plt.figure(figsize=(10, 6))
    plt.hist(til_sizes, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=mean_size, color='red', linestyle='--', label=f'Mean Distance: {mean_size:.2f}')
    plt.text(mean_size + 2, plt.ylim()[1] * 0.95, f'Mean: {mean_size:.2f}', color='black', fontsize=12, ha='left')
    
    min_til_size = np.min(til_sizes)
    plt.text(0.95, 0.95, f'Min: {min_til_size}', color='black', fontsize=12, ha='right', transform=plt.gca().transAxes)
    
    plt.xlabel('Number of Pixels')
    plt.ylabel('Frequency')
    plt.title('Histogram of Number of Pixels in each TIL')

    if save_path:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{image_name}_TILsize_histogram.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        plt.close()

    if show_plot:
        plt.show()
    else:
        plt.close()

    return mean_size, min_til_size, til_count


def image_to_features(mask_path: str,
                      original_image_path: str,
                      binary_flag: bool = True):
    """
    Generates the centroid coordinates and averaged RGB values
    for each TIL from a binary mask to create a feature matrix
    for DBSCAN
    
    Parameters
    -----
    mask_path (str): path to the TIL mask (binary or RGB). This will
    be used for calculating the centroid coordinates for each TIL.

    original_image_path (str): path to the original RGB patch.
    This will be used to extract the averaged RGB values for each TIL.

    binary_flag (bool): indicates whether or not averaged RGB values 
    are to be extracted as features. If False, then the feature matrix
    will be [X, Y, R, G, B] where (X,Y) are the centroid coordinates.
    If True, then the feature matrix is just [X, Y] and no averaged
    RGB values are calculated.

    Returns
    -----
    features (np.ndarray): an [n, 2] (binary_flag=True) or [n, 5] 
    (binary_flag=False) array of features for n TILs in the patch.

    binary_mask (np.ndarray): a 2D numpy array with 0's being background 
    and groups of 1's indicating TILs. 

    contours (list of np.ndarrays): a list where each element is a numpy
    array corresponding to the contour of a TIL. The order of contours
    corresponds to the feature matrix.
    """
    
    # reading image and turning it into a binary array
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(image.astype(np.uint8), 0.01, 1, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = 1D array where each element contains the [x,y] coords of pts along contour

    # List to store centroid coordinates
    centroids = []
    average_rgbs = []

    # Iterate through contours
    for contour in contours:
        # Calculate moments of the contour
        M = cv2.moments(contour)
        
        # Calculate centroid coordinates
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            centroids.append((centroid_x, centroid_y))

        if binary_flag is False: # calculates the avg RGB values for each ROI from the original patch
            # reading colored TIL mask
            image_bgr = cv2.imread(original_image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # otain the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
                # x,y: top left corner of bounding box
                # w,h: width, height of bounding box
        
            # create a bounding box around the TIL
            # roi = image_rgb[y-h:y+h, x-w:x+w]
            roi = image_rgb[y:y+h, x:x+w]

            # create contour mask based on bounding box and draw contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)
                # draws contour relative to bounding box coordinates

            # get averaged RGB values
            # mean_val = cv2.mean(roi)
            mean_val = cv2.mean(roi, mask=mask)
                # extracts mean values using pixels from the ROI 
                # according to to non-zero (white) pixels in the contour mask
            average_rgb = mean_val[:3] # ignores alpha
            average_rgbs.append(average_rgb)

    # convert centroids list to NumPy array
    centroids_array = np.array(centroids)

    # creating the feature matrix
    if binary_flag:
        features = centroids_array
    else:
        average_rgbs_array = np.array(average_rgbs)
        features = np.concatenate((centroids_array, average_rgbs_array), axis=1)
        
    return features, binary_mask, contours


def normalize_features(features: np.ndarray,
                       binary_flag: bool = True):
    """
    Normalizes the feature matrix using min-max scaling. The scales
    were deterined based on the dimensions of the patch (hard-coded)
    and the color scale (which ranges from 0-255)
    
    TODO: the normalization needs to be adjusted if different sized
    patches are used.

    Parameters
    -----
    features (np.ndarray): an [n, 2] (binary_flag=True) or [n,5] 
    (binary_flag=False) array of features for n TILs.

    binary_flag (bool): indicates whether or not averaged RGB values 
    are to be included as features. If False, then the feature matrix
    shold be [X, Y, R, G, B] where (X,Y) are the centroid coordinates.
    If True, then the feature matrix should be [X, Y].

    Returns
    -----
    features_norm (np.ndarray): array containing the normalized features
    using min-max scaling.
    """    
    # feature matrix should be [X, Y, R, G, B]
    # normalize the centroid coordinates
    features_norm = []

    if binary_flag:
        mins = np.array([0, 0])
        maxes = np.array([4000, 3000])
    else:
        mins = np.array([0, 0, 0, 0, 0])
        maxes = np.array([4000, 3000, 255, 255, 255])

    # do min-max normalization of the RGB values
    features_norm = (features - mins) / (maxes - mins)

    return features_norm


def calc_hyperparams(features: np.ndarray):
    """
    Determines the 'eps' hyperparameter based on the nearest
    neighbor distances of all the feature values. It is 
    calculated using the following equation:

        eps = mean_distance + 2 * stdev_distance

    note: For the centroid values, the feature input should be [X, Y]
    
    Parameters
    -----
    features (np.ndarray): an array of the feature to be used to calculate
    'eps' hyperparameter for DBSCAN. For example, this could be 
    just the (X,Y) coordinates.

    Returns
    -----
    eps (float): the 'eps' value to be used as a hyperparameter for
    the DSBCAN model to be fitted and predicted on the desired patch.

    nearest_distances (np.ndarray): an array where each element prepresents
    the distance to the nearest neighbor for that particular TIL. (should be
    the same length as the feature matrix)
    """ 
    # extract the centroid centers
    # til_centers = features[:, :2]

    # calculate pairwise distances between all centroid centers
    distances = distance.cdist(features, features)
    
    # replace diagonal distances (self-distance) with infinity
    # (to excluded these distances)
    np.fill_diagonal(distances, np.inf)
    
    # Find the minimum distance for each TIL
    nearest_distances = np.min(distances, axis=1)

    mean_distance = np.mean(nearest_distances)
    stdev_distance = np.std(nearest_distances)

    # determine 'eps' value for patch
    eps = mean_distance + 2 * stdev_distance

    return eps, nearest_distances


def cluster_processing(dbscan_labels: np.ndarray, 
                       plot: bool = False):
    """
    Extracts cluster label information from the DBSCAN model fitting and
    visualizes the number of clusters as well as how many pixels were
    assigned for each cluster. 

    Parameters
    -----
    dbscan_labels (np.ndarray): an array of each pixel's coordinates
    and the DBSCAN cluster label assigned to that pixel.
    
    plot (bool): True if a bar graph of the cluster labels and pixel
    count is desired.

    Returns
    -----
    cluster_labels_counts (dict): A dictionary of the cluster label
    counts where the key is the cluster label and the value is the 
    corresponding count.
    
    total_clusters (int): The total number of clusters identified by DBSCAN

    noise_percentage (float): the fraction of TILs classified as noise (i.e.
    given a -1 label)
    """
    
    # Extract unique labels and their counts
    unique_labels, label_counts = np.unique(dbscan_labels, return_counts=True)

    # Create a dictionary to store cluster labels and counts
    cluster_labels_counts = {}

    # Iterate over unique labels and counts
    for label, count in zip(unique_labels, label_counts):
        if label != -1:  # Exclude noise points
            cluster_labels_counts[label] = count

    # Extract cluster labels and counts from the dictionary
    labels = list(cluster_labels_counts.keys())
    counts = list(cluster_labels_counts.values())

    if plot: 
        # Plotting the bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')

        # Adding labels and title
        plt.xlabel('Cluster Label')
        plt.ylabel('Pixel Count')
        plt.title('Cluster Counts')

        # Adding text annotation for the total number of clusters
        total_clusters = len(labels)
        plt.text(0.05, 0.95, f'Total Clusters: {total_clusters}', transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top')

        # Display the plot
        plt.show()

        return cluster_labels_counts, total_clusters

    else:
        total_clusters = len(labels)

        # counting the total number of TILs counted
        total_count = 0
        for count in cluster_labels_counts.values():
            total_count += count

        # counting the number of TILs classified as noise
        noise_percentage = (1 - total_count / 1335) * 100

        print(cluster_labels_counts)
        print(f"Total TILs Clustered: {total_count}")
        print(f"Total # Clusters: {total_clusters}")
        print(f"Noise Points: {noise_percentage:.2f}%")

        return cluster_labels_counts, total_clusters, noise_percentage


def dbscan_calculations(spatial_weight: float, 
                        features: np.ndarray, 
                        rgb_calc: str):
    """
    Performs the calculations for DBSCAN prior to fitting and predicting
    the model. This pre-computes the euclidean distances for the 
    centroid coordinates and either the euclidean, cosine, or delta_e
    distances for the RGB values.
    
    Parameters
    -----
    spatial_weight: A number given between 0-1 indicating the weight 
    given to the centroid coordinates vs. the averaged RGB values. 
    The RGB weight is simply equal to 1 - spatial_weight.
    
    features (np.ndarray): an [n, 5] array of the features (X, Y, R, G, B) 
    to be used for DBSCAN. 

    rgb_calc (str): Indicates the distance calculation method to be used 
    for the averaged RGB values ('euclidean', 'cosine', or 'delta_e').
    
    TODO: cosine portion is currently broken
    TODO: no option given if no RGB values are given (i.e. no binary_flag)

    Returns
    -----
    combined_distances (np.ndarray): an array of the linear combination 
    of the distance calculations for the centroid coordinates and
    RGB values.
    
    spatial_distances (np.ndarray): an array of the euclidean distances 
    for the centroid coordinates.
    
    color_distances (np.ndarray): an array of the distance calculations 
    for the RGB values.
    """ 
    # assuming your feature matrix is the following:
    # [X, Y, R, G, B] where (X,Y) are the centroid coordinates
    # and R, G, B are the averaged values for each contour

    # define weights for spatial distance and RGB 
    rgb_weight = 1 - spatial_weight

    # calculate pairwise distances for spatial coordinates
    spatial_distances = euclidean_distances(features[:, :2])  
        # Assuming first two columns are spatial coordinates

    if rgb_calc == 'euclidean': # calculate euclidean distances or similarities for RGB values
        color_distances = euclidean_distances(features[:, 2:])  # Assuming remaining columns are RGB values
        
    elif rgb_calc == 'cosine': # calculate cosine distances for RGB values
        color_distances = 1 - cosine_similarity(features[:, 2:])
        
    elif rgb_calc == 'delta_e': # converts RGB to LAB and computes distances
        # convert RGB to LAB
        lab_values = color.rgb2lab(features[:, 2:].astype(np.uint8))
        color_distances = distance.cdist(lab_values, lab_values, metric=color.deltaE_ciede2000)

    else:
        raise ValueError("calc parameter must be either 'euclidean', 'cosine', or 'delta_e'")
    
    # linear combination of spatial and RGB info
    combined_distances = spatial_weight * spatial_distances + rgb_weight * color_distances

    return combined_distances, spatial_distances, color_distances


def visualize_clusters(dbscan_labels: np.ndarray, 
                       mask: np.ndarray, contours: list, 
                       original_img: np.ndarray,
                       dbscan_hyperparam: dict, 
                       total_clusters: int,
                       save_image: bool = False, out_path: str = None,
                       filename: str = None
                       ):
    """
    Takes the DBSCAN output and (1) plots the contours and (2) creates 
    an overlay over the original patch. If save_image=True, the images are 
    saved in the output path with dpi=600.
    
    Parameters
    -----
    dbscan_labels (np.ndarray): The DBSCAN labels extracted from the 
    fit and predicted model on the patch. Each label represents the 
    cluster each TIL belongs to.
    
    mask (np.ndarray): the numpy array containing the binary or RGB mask of
    the original image.
    
    contours (list of np.ndarrays): a list where each element is a numpy
    array corresponding to the contour of a TIL. The order of contours
    corresponds to the feature matrix. 

    original_img (np.ndarray): the original patch (RGB) as a numpy array.
    
    dbscan_hyperparam (dict): ihe dictionary of hyperparameters that were
    used for fitting and predicting the DBSCAN model. The format should be
    {'eps': x, 'min_samples': y} for some x and y.
    
    total_clusters (int): a number indicating the number of unique clusters
    identified by the DBSCAN model.
    
    save_image (bool): indicates whether the plot and overlay
    should be saves (True) or not (False). The default is False. If True,
    the out_path and filename must be provided.
    
    out_path (str): the path where the plot and overlay will be saved.
    
    filename (str): the name of the patch that includes the relevant
    information (i.e. patient number, patch position, etc.).

    Returns
    -----
    N/A
    """ 
    unique_labels = set(dbscan_labels)
    # core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
    # core_samples_mask[dbscan.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))] # create color map
    overlay = original_img.copy()

    # ensure shape of plot = aspect ratio of original image/mask
    aspect_ratio = mask.shape[1] / mask.shape[0] 
    fig_width = 20
    fig_height = fig_width / aspect_ratio

    # creating figure for cluster contours
    if save_image: # this is because showing a 600 dpi image takes a lot of time
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=600)  # Set figure size and dpi for high resolution
    else: # for faster processing if you just want to look at the results
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # loop through DBSCAN labels to (1) plot contours + (2) draw them on the original image
    for k, color in zip(unique_labels, colors): # loop through each dbscan label
        if k == -1: # color noise points as black
            color = [0, 0, 0, 1]

        class_member_mask = dbscan_labels == k # create boolean mask for labels
        label_indices = np.where(class_member_mask)[0]  # find indices where mask is True 

        for i in label_indices: # (1) draw contours
            contour = contours[i]
            xy = np.squeeze(contour)  # convert contour to (N, 2) array
            ax.plot(
                xy[:, 0],
                -xy[:, 1],  # invert y-coordinates (unsure why I have to do this)
                "-",
                color=color,
                linewidth=1,
            )
            
            # (2) draw contours on original image
            bgr_color = tuple((color[0] * 255, color[1] * 255, color[2] * 255))
            cv2.drawContours(overlay, [contour.astype(int)], -1, bgr_color, thickness=4)

    # image of clustering plot
    ax.set_title(f"{filename} - 'eps': {dbscan_hyperparam['eps']}, 'min_samples': {dbscan_hyperparam['min_samples']}, number of clusters: {total_clusters}")
    ax.axis('off')

    # (1) saving clustering contour plot
    if save_image:
        plt.savefig(os.path.join(out_path, f"{filename}_DBSCAN.jpg"), dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # plt.figure(figsize=(fig_width, fig_height))
    #     # use the above line if you want to speed up the image generation in a .ipynb
    # # plt.figure(figsize=(fig_width, fig_height), dpi=600)
    # plt.imshow(overlay)
    # plt.title(f"{filename} - 'eps': {dbscan_hyperparam['eps']}, 'min_samples': {dbscan_hyperparam['min_samples']}, number of clusters: {total_clusters}")
    # plt.axis('off')

    # (2) saving contour overlay with original patch
    if save_image: # saving with high resolution (time consuming)
        plt.figure(figsize=(fig_width, fig_height), dpi=600)
        plt.imshow(overlay)
        plt.title(f"{filename} - 'eps': {dbscan_hyperparam['eps']}, 'min_samples': {dbscan_hyperparam['min_samples']}, number of clusters: {total_clusters}")
        plt.axis('off')
        overlay_image_path = os.path.join(out_path, f"{filename}_DBSCAN_overlay.jpg")
        cv2.imwrite(overlay_image_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR
        plt.close()
    else: # if not saving image, loads lower resolution image faster
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(overlay)
        plt.title(f"{filename} - 'eps': {dbscan_hyperparam['eps']}, 'min_samples': {dbscan_hyperparam['min_samples']}, number of clusters: {total_clusters}")
        plt.axis('off')


def sbm_dbscan_wrapper(mask_path: str, original_img_path: str, out_path: str = None, 
                       hyperparameter_dict: dict = {'eps': 150,'min_samples': 100},
                       multiple_patches_flag: bool = False,
                       features_extracted: list = None,
                       binary_flag: bool = True, spatial_wt: float = None, rgb_calc: str = None, 
                       cluster_plot: bool = False, save_image: bool = False):
    """
    Takes in the TIL mask, creates the necessary feature matrix, 
    and trains a DSBCAN model based on those features and hyperparameters
    given the paths to the original image and the corresponding mask. 
    Depending on the inputs, can also cluster binary/RGB images, 
    customize the distance calculations for the RGB features,
    generate a contour plot + overlay, as well as save the clustered
    images with high resolution (600 dpi) if desired.
    
    Parameters
    -----
    mask_path (str): If multiple_patches_flag=True, this should
    be the folder path containing all of the TIL masks. If it is false, it
    should be the file path to the file itself.

    original_img_path (str): If multiple_patches_flag=True, this should
    be the folder path containing all of the images of the original patches. 
    If it is false, it should be the file path to the file itself. 
    (note: the name of the original image should match the filename of 
    the corresponding TIL mask)

    out_path (str): Path to a folder where all the images/results will
    be saved

    hyperparameter_dict (dict): Dictionary with DBSCAN hyperparameters ('eps'
    and 'min_samples') as the keys and their corresponding values.
    
    multiple_patches_flag (bool): Indicates whether you need to convert
    a folder of patches or a singular patch to binary masks.

    features_extracted (list): If the feature extraction has already been done 
    separately, the user must input a list: [features, binary_mask, contours]
    corresponding to the outputs from the image_to_features function above.
    Otherwise, the feature extraction will be automatically be done if no 
    input is given.
    (note: this step usually takes the longest) 

    binary_flag (bool): Inidcates whether the TIL mask to be clustered
    is binary (True) or in RGB (False). If it is set to False, the following
    parameters must also be given: spatial_wt, rgb_calc.
    
    spatial_wt (float): A number between 0 and 1 indicating the weight of
    the spatial coordinates vs. the RGB values in the feature matrix
    (if binary_flag=False)
    
    rgb_calc (str): If the binary_flag=False, this specifies what
    matric to use to calculate the differences in the averaged RGB values.
    
    cluster_plot (bool): True if you want to plot and visualize the DBSCAN
    clustering results in line (i.e. if you're running the script in a 
    .ipynb file).
    
    save_image (bool): True if you want to save the clustering results
    as a .jpg file (600 dpi).

    Returns
    -----
    results_dict (dict): Dictionary where each key is the image file name
    (i.e. x.tif) and the corresponding values are the DBSCAN labels (np.ndarray), 
    number of unique clusters (int), and the percentage of noise (float) as a 
    result of the fitted and predicted model formatted as:
         {'xxx.tif': [dbscan_labels, total_clusters, noise]}
    
    features_dict (dict): Dictionary where each key is the image file name
    and the corresponding value is a list of the feature matrix (np.ndarray), 
    binary mask (np.ndarray), and contours (list) formatted as: 
        {'xxx.tif': [features, binary_mask, contours]}
        * note: [features, binary_mask, contours] is the output of image_to_features
    """
    
    # initialize output dictionaries
    results_dict = {} # contains the labels, total clusters, and noise from the clustering for each patch
    features_dict = {} # contains the features, binary_mask, contours for each patch

    if multiple_patches_flag:
        for filename in tqdm(os.listdir(mask_path)):
            if filename.endswith(".tif"): # looping over each patch
                
                ## store info for each patch
                patch_cluster_info = {}

                mask = os.path.join(mask_path, filename)
                original_img_file = os.path.join(original_img_path, filename)
                original_img_bgr = cv2.imread(original_img_file)
                original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
                    # make sure the filenames for the mask and original patch match in each folder
                
                ## creating the feature matrix
                if features_extracted:
                    features = features_extracted[filename][0]
                    binary_mask = features_extracted[filename][1]
                    contours = features_extracted[filename][2]
                else:
                    features, binary_mask, contours = image_to_features(mask, 
                                                                        original_img_file, 
                                                                        binary_flag)
                
                ## creating the hyperparameter dict if none is given
                if hyperparameter_dict is None:
                    eps, _ = calc_hyperparams(features)
                    hyperparam = {'eps': eps, 'min_samples': 20}
                else:
                    hyperparam = hyperparameter_dict

                ## DBSCAN Model Fitting
                if rgb_calc:
                    dbscan_features, _, _ = dbscan_calculations(spatial_wt, 
                                                           features, 
                                                           rgb_calc)
                    dbscan = DBSCAN(**hyperparam,
                                    metric='precomputed').fit(dbscan_features)
                    dbscan_labels = dbscan.labels_
                
                else: 
                    dbscan_features = features
                    dbscan = DBSCAN(**hyperparam).fit(dbscan_features)
                    dbscan_labels = dbscan.labels_

                # calculate the total number of clusters output by DBSCAN
                _, total_clusters, noise = cluster_processing(dbscan_labels, 
                                                       plot=False)
                
                # assigning each filename the clustering labels 
                # and total number of clusters
                patch_cluster_info['labels'] = dbscan_labels
                patch_cluster_info['total_clusters'] = total_clusters
                patch_cluster_info['noise'] = noise

                results_dict[filename] = patch_cluster_info

                if cluster_plot:
                    visualize_clusters(dbscan_labels, 
                                       binary_mask, 
                                       contours, 
                                       original_img_rgb,
                                       hyperparam, 
                                       total_clusters,
                                       save_image,
                                       out_path,
                                       filename
                                       )

    else: # if only a single patch is given

        filename = os.path.splitext(os.path.basename(mask_path))[0]
        # read in original patch using OpenCV
        original_img_bgr = cv2.imread(original_img_path)
        # convert BGR to RGB channels
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)

        ## creating the feature matrix
        if features_extracted: # if feature extraction has already been done
            # features extracted = [features, binary_mask, contours]
            features = features_extracted[0]
            binary_mask = features_extracted[1]
            contours = features_extracted[2]

        else:
            # generate feature matrix from TIL mask 
            features, binary_mask, contours = image_to_features(mask_path, original_img_path, binary_flag)
            features_dict[filename] = [features, binary_mask, contours]

        ## creating the hyperparameter dict if none is given
        if hyperparameter_dict is None:
            eps, _ = calc_hyperparams(features)
            hyperparam = {'eps': eps, 'min_samples': 20}
        else:
            hyperparam = hyperparameter_dict

        ## DBSCAN Model Fitting
        if rgb_calc: # only relevant if binary_flag=False
            dbscan_features, _, _ = dbscan_calculations(spatial_wt, 
                                                    features, 
                                                    rgb_calc)
            dbscan = DBSCAN(**hyperparam,
                            metric='precomputed').fit(dbscan_features)
            dbscan_labels = dbscan.labels_
        
        else: 
            dbscan_features = features
            dbscan = DBSCAN(**hyperparam).fit(dbscan_features)
            dbscan_labels = dbscan.labels_

        ## calculate the total number of clusters output by DBSCAN
        _, total_clusters, noise = cluster_processing(dbscan_labels, 
                                                plot=False)
        
        ## assigning each filename the clustering labels 
        # and total number of clusters
        results_dict['labels'] = dbscan_labels
        results_dict['total_clusters'] = total_clusters
        results_dict['noise'] = noise

        ## generating contour plot + overlay
        if cluster_plot:
            print('creating plot')
            visualize_clusters(dbscan_labels, 
                                binary_mask, 
                                contours,
                                original_img_rgb,
                                hyperparam, 
                                total_clusters,
                                save_image,
                                out_path,
                                filename,
                                )
            
    return results_dict, features_dict


def export_dict_to_csv_nn(filename, dct):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Mean Distance', 'Max Distance', "Std Dev Distance"])
        for key, value in dct.items():
            writer.writerow([key, value['mean_distance'], value['max_distance'], value['stdev_distance']])


def export_dict_to_csv_ts(filename, dct):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Mean Size', 'Minimum TIL Size', 'TIL Count'])
        for key, value in dct.items():
            writer.writerow([key, value['mean_size'], value['min_til_size'], value['til_count']])


def extract_cluster_label_counts(tiff_path):
    # Load the TIFF image
    cluster_labels_image = tifffile.imread(tiff_path)

    # Flatten the image into a 1D array
    flattened_labels = cluster_labels_image.flatten()

    # Count the occurrences of each unique label
    unique_labels, label_counts = np.unique(flattened_labels, return_counts=True)

    # Create a dictionary to store the label counts
    label_counts_dict = dict(zip(unique_labels, label_counts))

    return label_counts_dict
