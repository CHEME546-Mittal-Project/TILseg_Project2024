# Internal Imports
import os

# External Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from skimage.measure import regionprops, label
import sklearn.cluster
import tifffile

# Local Imports
from tilseg.refine_kmeans import km_dbscan_wrapper


def sbm_to_binarymask(in_path: str,
                      multiple_patches_flag: bool = False):
    """
    Converts a .tif file(s) into a binary mask(s) of 1's and 0's 

    Parameters
    -----
    in_path (str): If the multiple_patches_flag is true, this should
    be the folder path containing all of the images. If it is false, this
    should be the file path to the image itself.
    multiple_patches_flag (bool): Indicates whether you need to convert
    a folder of patches or a singular patch to binary masks.
    
    Returns
    -----
    binary_masks_dict (dict): If the multiple_patches_flag is true, a dictionary 
    will be created with the file names as the keys and the corresponding 
    binary 2D arrays as the values.
    binary_mask (np.ndarray): If the multiple_patches_flag is false, a singular 
    binary 2D array will be created where 0's are background and 1's are 
    pixels with color.
    """
    binary_masks_dict = {}
    #TODO binary_mask_dict = {} also output a dict for a single patch?

    if multiple_patches_flag:
        for filename in os.listdir(in_path):
            if filename.endswith(".tif"):
                # Read the image (assuming in_path is a folder path)
                image_path = os.path.join(in_path, filename)
                patch_image = cv2.imread(image_path)

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)

                # Threshold the grayscale image to create the binary mask
                _, binary_mask = cv2.threshold(gray_image, 1, 1, cv2.THRESH_BINARY)

                # Normalize the binary mask
                cv2.normalize(binary_mask, 
                              binary_mask, 
                              alpha=0, 
                              beta=1, 
                              norm_type=cv2.NORM_MINMAX)
                # Add the binary mask to the dictionary with the image name as the key
                binary_masks_dict[filename] = binary_mask

        return binary_masks_dict

    else: 
        # Read the image (assuming in_path is a file path)
        patch_image = cv2.imread(in_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to create the binary mask
        _, binary_mask = cv2.threshold(gray_image, 1, 1, cv2.THRESH_BINARY)

        # Normalize the binary mask
        cv2.normalize(binary_mask, binary_mask, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
        return binary_mask


def image_to_features(image_path: str,
                     binary_flag: bool = True):
    """
    Generates the spatial coordinates from a binary mask as features 
    to cluster with DBSCAN
    
    Parameters
    -----
    mask (np.ndarray): a mask with 1's corresponding to the pixels 
    involved in the TIL cluster (i.e. with the most contours) and 0's for pixels not
    binary flag (bool): True if the mask is binary which should be a 2D numpy array 
    consisting of only 1's (i.e. pixels) and 0's (i.e. background). False if the 
    mask is colored in which it should be a 3D numpy array containing the RGB 
    values of an image.

    Returns
    -----
    features (np.array) is a an array where each row corresponds to a set of 
    coordinates (x,y) of the pixels where the mask had a value of 1. If the 
    binary_flag was set to False, then the feature matrix should additionally 
    include columns corresponding to the RGB values of the pixel.
    """
    
    if binary_flag:
        # reading image and turning it into a binary array
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(image.astype(np.uint8), 0.01, 1, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours = 1D array where each element contains the [x,y] coords of pts along contour

        # List to store centroid coordinates
        centroids = []

        # Iterate through contours
        for contour in contours:
            # Calculate moments of the contour
            M = cv2.moments(contour)
            
            # Calculate centroid coordinates
            if M['m00'] != 0:
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
                centroids.append((centroid_x, centroid_y))

        # Convert centroids list to NumPy array
        centroid_coords = np.array(centroids)
        
    # else:
        

    return centroid_coords, binary_mask, contours


def nearest_neighbor_distance(binary_mask, 
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
    cluster_centers = [region.centroid for region in regions]
    
    # Calculate pairwise distances between all cluster centers
    distances = distance.cdist(cluster_centers, cluster_centers)
    
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

        return cluster_labels_counts, total_clusters


def sbm_dbscan_wrapper(in_path: str, 
               out_path: str = None, 
               hyperparameter_dict: dict = {'eps': 150,'min_samples': 100}, 
               multiple_patches_flag: bool = False,
               binary_flag: bool = True,
               save_image: bool = False, 
               cluster_plot: bool = False,
               nn_plot: bool = False):
    """
    Takes in the TIL mask, creates the necessary feature matrix, 
    and trains a DSBCAN model based on those features and the given 
    hyperparameters.
    
    Parameters
    -----
    in_path (str): If the multiple_patches_flag is true, this should
    be the folder path containing all of the images. If it is false, this
    should be the file path to the image itself.
    out_path (str): Path to a folder where all the images/results will
    be saved
    hyperparameter_dict (dict): Dictionary with DBSCAN hyperparameters ('eps'
    and 'min_samples') as the keys and their corresponding values.
    multiple_patches_flag (bool): Indicates whether you need to convert
    a folder of patches or a singular patch to binary masks.
    save_image (bool): True to save a high resolution image (600 dpi)
    of the DBSCAN clustering results.
        # TODO still need to fix this in km_dbscan_wrapper
    cluster_plot (bool): True to calculate the nearest neighbor distances 
    between all TILs.
        # TODO: need to equate this to something? maybe add to one of the output dicts
    nn_plot (bool): True to calculate the pixel are of each TIL.
        # TODO: need to equate this to something? maybe add to one of the output dicts

    Returns
    -----
    labels_dict (dict): Dictionary where each key is the image file name
    and the corresponding values are the DBSCAN labels for that image. 
    clusters_dict (dict): Dictionary where each key is the image file name
    and the corresponding value if the total number of clusters for that image.
    """
    
    # initialize dictionaries
    labels_dict = {}
    clusters_dict = {}

    if multiple_patches_flag:
        for filename in os.listdir(in_path):
            if filename.endswith(".tif"): # looping over each patch
                
                # generate feature matrix
                image_path = os.path.join(in_path, filename)
                centroid_coords, binary_mask = image_to_features(image_path, binary_flag)

                # DBSCAN Model Fitting
                dbscan = sklearn.cluster.DBSCAN(**hyperparameter_dict)
                dbscan_labels = dbscan.fit_predict(centroid_coords)

                # assigning DBSCAN labels to TILs based on centroids
                colors = np.zeros_like(binary_mask)
                for i, label in enumerate(dbscan_labels):
                    centroid_x, centroid_y = centroid_coords[i]
                    colors[centroid_y, centroid_x] = label
                
                # saving image
                if save_image:
                    plt.savefig(os.path.join(out_path, f"{filename}_DBSCAN.jpg"), dpi=600, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                
                # calculate the total number of clusters output by DBSCAN
                _, total_clusters = cluster_processing(dbscan_labels, 
                                                           plot=cluster_plot)
                
                # assigning each filename the clustering labels 
                # and total number of clusters
                labels_dict[filename] = dbscan_labels
                clusters_dict[filename] = total_clusters

                if cluster_plot:
                    til_size(mask)
                if nn_plot:
                    nearest_neighbor_distance(mask)
       
        return labels_dict, clusters_dict

    # else: 
    #     dbscan_labels, dbscan_model = km_dbscan_wrapper(binary_mask, 
    #                                             hyperparameter_dict, 
    #                                             save_filepath = out_path,
    #                                             binary_flag = True,
    #                                             save_image = save_image)
        
    #     cluster_labels_counts, total_clusters = cluster_processing(dbscan_labels, 
    #                                                plot = cluster_plot)    
        
    #     # get file name from in_path
    #     filename = os.path.basename(in_path)
    
    #     labels_dict[filename] = dbscan_labels
    #     clusters_dict[filename] = total_clusters

    #     if cluster_plot:
    #         til_size(binary_mask)
    #     if nn_plot:
    #         nearest_neighbor_distance(binary_mask)       
        
    #     return labels_dict, clusters_dict
    
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
