"""
An image preprocessing module that can take in an svs
file as an image and return separate patches of that
image broken up and filtered down to hold patches of
only one type. This is then used for testing/using a
machine learning model or for superpatch creation in
a consequtive module.
"""

from skimage import io

import collections
import math
import openslide
import os
import scipy
import uuid

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def open_slide(slidepath):
    """
    A function that opens a slide and returns an
    OpenSlide object and the slide's dimensions.

    Parameters
    -----
    slidepath: the complete path to the slide file (.svs)

    Returns
    -----
    slide: the slide object created by OpenSlide
    slide_x: the x dimension of the slide
    slide_y: the y dimension of the slide
    """

    # get the slide object
    slide = openslide.OpenSlide(slidepath)

    # get x and y dimensions
    slide_x, slide_y = slide.dimensions

    return slide, slide_x, slide_y


def get_tile_size(maximum, size, cutoff=4):
    """
    A function that takes in a slide dimension and returns
    the optimal breakdown of each slide into x patches.

    Parameters
    -----
    maximum: the maximum dimension desired
    size: the size of the entire slide image
    cutoff: the maximum number of pixels to remove (default is 4)

    Returns
    -----
    dimension: the desired pixel size needed
    slices: the number of slices needed in the given direction
    remainder: the number of pixels lost with the slicing provided
    """

    # iterate through possible sizes of tiles starting
    # with the largest possible tile size
    for dimension in reversed(range(0, (maximum + 1))):

        # calculate the remainder (number of pixels missing)
        remainder = size % dimension

        # check if the remainder is less than cutoff
        if remainder <= cutoff:

            # calculate the number of patches made
            slices = math.trunc(size / dimension)

            # return requested values
            return dimension, slices, remainder

        # if not, continue to the next value for tile size
        else:
            continue


def percent_of_pixels_lost(lost_x, patch_x, lost_y, patch_y, x_size, y_size):
    """
    A function that calculates the total percentage of pixels
    lost from the whole slide when the slicing occurs.

    Parameters
    -----
    lost_x: the number of pixels lost in the x direction
    patch_x: the number of patches that are split in the x direction
    lost_y: the number of pixels lost in the y direction
    patch_y: the number of patches that are split in the y direction
    x_size: the total number of pixels in the x direction of the slide
    y_size: the total number of pixels in the y direction of the slide

    Returns
    -----
    percent: the percent of pixels deleted, rounded to two places
    """

    # calculate the percent
    percent = (lost_x * patch_x + lost_y * patch_y - lost_x * lost_y) \
        / (x_size * y_size) * 100

    return percent


def save_image(path, name, image_array):
    """
    A function that saves an image given a path.

    Parameters
    -----
    path: the complete path to a directory to which the image should be saved
    name: the name of the file, with extension, to save
    image_array: a numpy array that stores image information
    """

    # create the entire saving directory
    save_as = os.path.join(path, name)

    # save the image
    io.imsave(save_as, image_array, check_contrast=False)

    return


def create_patches(slide, ypatch, xpatch, xdim, ydim):
    """
    A function that creates patches and yields an numpy
    array that describes the image patch for each patch
    in the slide.

    Parameters
    -----
    slide: the OpenSlide object of the entire slide
    ypatch: the dimension of the patch in the y direction
    xpatch: the dimension of the patch in the x direction
    xdim: the size of the patch in the x direction
    ydim: the size of the patch in the y direction

    Returns
    -----
    np_patches: a list of all patches, each as a number array
    patch_position: a list of tuples containing indices
    """

    # establish an empty patches list that will contain all patches
    np_patches = []

    # establish an empty list that will contain tuples of positions
    patch_position = []

    # iterate through the n x patches that will be made
    for xpatches in range(1, xpatch + 1):

        # get the starting left x coordinate of the patch
        start_x = (xpatches - 1) * xdim

        # iterate through the m y patches that will be made
        for ypatches in range(1, ypatch + 1):

            # get the starting left y coordinate of the patch
            start_y = (ypatches - 1) * ydim

            # convert patch into np array
            npimage = np.asarray(slide.read_region((start_x, start_y), 0, (xdim, ydim)))

            # reformat array so it can be read properly
            np_patch = np.array(npimage)[:,:,:3]

            # append new patch to the master list of patches
            np_patches.append(np_patch)

            # append position to patch position list
            patch_position.append((xpatches, ypatches))

    return np_patches, patch_position


def get_average_color(img):
    """
    A function that returns the average RGB color
    of an input image array (in this case a patch).

    Parameters
    -----
    img: a numpy array containing all information about the RGB colors in a patch

    Returns
    -----
    average: a numpy array containing the RGB code for the average color
        of the entire patch
    """

    # calculate the average
    average = img.mean(axis=0).mean(axis=0)

    return average


def get_grey(rgb):
    """
    A function that calculates the greyscale
    value of an image given an RGB array.

    Parameters
    -----
    rgb: a numpy array containing three values, one each for R, G, and B

    Returns
    -----
    grey: the greyscale value of an image/patch
    """

    grey = (rgb[0] + rgb[1] + rgb[2]) / 3

    return grey


def save_all_images(df, path, f):
    """
    A function to save all the images as background or tissue.

    Parameters
    -----
    df: the dataframe that is already created containing patches,
        average patch color, and the greyscale value
    path: the path to which the folders and subdirectories will be made
    f: the slide .svs file that is currently being read

    Returns
    -----
    None, but all images are saved
    """

    # check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError('The file path given does not exist.')
    else:
        pass

    # check if the file name has an extension
    if '.' not in f:
        raise TypeError('The file name provided for the image has no extension.')
    else:
        pass

    # check that dataframe is actually a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The dataframe entered is not a dataframe object.')
    else:
        pass

    # check that the dataframe has a position column
    if 'patch_xy' not in df.columns:
        raise ValueError('The dataframe does not have positions shown \
                          under a patch_xy column for each patch.')
    else:
        pass

    # check that the dataframe column patch_xy only contains tuples
    for elem in df.patch_xy:
        if not isinstance(elem, tuple):
            raise TypeError('The position column does not contain only tuples.')
        else:
            pass

    # get name of file without extension
    slide_name = f.split('.')[0]

    # name all used directories
    slide_name_path = os.path.join(path, slide_name)

    # check if the folder does not already exist
    assert not os.path.isfile(slide_name_path), 'An existing \
        folder with the slide name already exists.'

    # make all necessary directories
    os.mkdir(slide_name_path)

    # iterate through all rows of the dataframe
    for index, row in df.iterrows():

        # name the file that will be saved based on its index on the whole slide image
        name = 'position_'+str(row['patch_xy'][0])+'_'+str(row['patch_xy'][1])+'tissue.tif'

        # save the image
        save_image(slide_name_path, name, row['patches'])

    return


def find_max(arr, cutoff, greater):
    """
    A function that finds the max value of a list/array
    within a specific range.

    Parameters
    -----
    arr: the array that contains the list of data in question
    cutoff: the value at which you want to start looking for a maximum
    greater: a boolean that determines if you want the maximum above
        or below the cutoff (above is when greater=False)

    Returns
    -----
    loc: the index (from zero) at which the maximum value occurs
    """

    # check that greater is a boolean
    if not isinstance(greater, bool) or (greater != True and greater != False):
        raise TypeError('The greater argument must be True or False.')
    else:
        pass

    # check that arr is a list or array
    if not isinstance(arr, (collections.abc.Sequence, np.ndarray)):
        raise TypeError('The input list must be an array or list.')
    else:
        pass

    # check that the cutoff value is an integer or float
    if not isinstance(cutoff, (int, float)):
        raise TypeError('The cutoff value must be an integer or float value.')
    else:
        pass

    # check that all list values are positive
    if any(item < 0 for item in arr):
        raise ValueError('The list can only contain non-negative values.')
    else:
        pass

    # a dummy number for the max that will never actually be the max
    maximum = 0

    # iterate through the array, but enumerate so that it is easy to get index
    for index, number in enumerate(arr):

        # if interested in a maximum below the cutoff and the
        # index is greater than this cutoff, then break out of the loop
        if greater and index > cutoff:
            break

        # if interested in a maximum above the cutoff and the index
        # is less than the cutoff, continue looping but do not do anything
        if not greater and index < cutoff:
            continue

        # check if the number in the appropriate range is greater than the maximum
        if number > maximum:

            # if it is, reassign the maximum value at this new value and record the index
            maximum = number
            loca = index

        # if the number is not greater than the maximum do nothing and continue
        else:
            continue

    return loca


def find_min(arr, range_min, range_max):
    """
    A function that finds the min value of a list/array
    within a specific range.

    Parameters
    -----
    arr: the array that contains the list of data in question
    range_min: the lower bound on which to look for the minimum
    range_max: the upper bound on which to look for the minimum

    Returns
    -----
    loc: the index (from zero) at which the minimum value occurs
    """

    # check that the range_max value is an integer or float
    if not isinstance(range_max, (int, float)):
        raise TypeError('The range_max value must be an integer or float value.')
    else:
        pass

    # check that the range_min value is an integer or float
    if not isinstance(range_min, (int, float)):
        raise TypeError('The range_min value must be an integer or float value.')
    else:
        pass

    # check that arr is a list or array
    if not isinstance(arr, (collections.abc.Sequence, np.ndarray)):
        raise TypeError('The input list must be an array or list.')
    else:
        pass

    # check that all list values are positive
    if any(item < 0 for item in arr):
        raise ValueError('The list can only contain non-negative values.')
    else:
        pass

    # check that the range min and range max are less than or greater than
    assert range_min < range_max, 'The range minimum is greater than the maximum.'
    assert range_min != range_max, 'The range minimum and maximum are the same.'

    # a dummy number for the min that will never actually be the min
    minimum = 1000000

    # iterate through the array, but enumerate so that it is easy to get index
    for index, number in enumerate(arr):

        # check if the index is between the desired range
        if index > range_min and index < range_max:

            # if it is in the correct range then check if the number
            # is less than the current minimum
            if number < minimum:

                # if it is less than the current minimum, reassign
                # the minimum and record the new index
                minimum = number
                loca = index

            # if it is in the correct range but not less than the 
            # current minimum then continue through the loop and do nothing
            else:
                continue

        # if the index is out of the desired range, continue and do nothing with that index
        else:
            continue

    return loca


def compile_patch_data(slide, ypatch, xpatch, xdim, ydim):
    """
    A function that compiles all relevant data for all patches into a dataframe.

    Parameters
    -----
    slide: the OpenSlide object of the entire slide
    ypatch: the number of patches in the y direction
    xpatch: the number of patches in the x direction
    xdim: the size of the patch in the x direction
    ydim: the size of the patch in the y direction

    Returns
    -----
    patchdf: a pandas dataframe containing the three following
    """

    # create a dataframe to contain all patch information from a slide
    patchdf = pd.DataFrame()

    # unpack information from created patches
    patch_list, index_list = create_patches(slide, ypatch, xpatch, xdim, ydim)

    # create a column with a numpy array patch in each row of the dataframe
    patchdf['patches'] = patch_list

    # create a column with a tuple indicating the corresponding patch
    patchdf['patch_xy'] = index_list

    # for each patch, calculate the average RGB value of the entire patch
    patchdf['RGB_avg'] = patchdf.apply(lambda row: get_average_color(row['patches']), axis=1)

    # add another column that converts the average RGB color to a greyscale color
    patchdf['greys'] = patchdf.apply(lambda row: get_grey(row['RGB_avg']), axis=1)

    return patchdf


def is_it_background(cutoff, actual):
    """
    A function that tests if a specific image should
    be classified as a background image or not.

    Parameters
    -----
    cutoff: the cutoff value for a background image

    Returns
    -----
    background: a boolean that is True if the patch should be considered background
    """

    # test if the actual value is greater than the cutoff
    if actual > cutoff:

        # if it is, then background is set to True
        background = True

    # if not, the background is set to False
    else:
        background = False

    return background


def sort_patches(df, lin_space=100, approx_between=200):
    """
    A function that starts sorting patches based on a KDE,
    determines a cutoff value, and calculates the final
    dataframe for each image.

    Parameters
    -----
    df: the dataframe that is already created containing patches,
        average patch color, and the greyscale value
    lin_space: the multiple by which the KDE axis will be split into
        while it is being formed for a PDF (default is 100)
    approx_between: the approximate value at which the grey values
        will be split into two populations in the bimodal distribution.
        This is usually around 200 for slides and is going to be
        set to that as a default.

    Returns
    -----
    df: an updated dataframe with a background column that indicates
        if a patch should be considered background or not
    """

    # check that the input is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('The input dataframe is not a dataframe.')
    else:
        pass

    # check that the dataframe contains a greys column
    if 'greys' not in df.columns:
        raise KeyError('The input dataframe does not contain a greys column.')
    else:
        pass

    # check that the dataframe column greys only contains numeric values
    for elem in df.greys:
        if not isinstance(elem, (int, float)):
            raise TypeError('The position column does not contain only \
                            numeric values.')
        else:
            pass


    # calculate min, max, and range of grey values
    minimum_grey = int(df['greys'].min())
    maximum_grey = int(df['greys'].max())
    range_grey = maximum_grey - minimum_grey

    # put all grey values into a list
    list_of_greys = df['greys'].values.tolist()

    # create a linspace for all grey values for which the PDF will be calculated
    grey_space = np.linspace(minimum_grey, maximum_grey, range_grey * lin_space)

    # create a KDE distribution from the list of greys
    kde_distr = scipy.stats.gaussian_kde(list_of_greys)

    # use the KDE distribution to create a PDF of the grey values along the grey space
    kde_pdf = kde_distr(grey_space)

    # find all local maxima and minima
    color_max = find_max(kde_pdf, (approx_between - minimum_grey) * lin_space, True) 
    background_max = find_max(kde_pdf, (maximum_grey - approx_between) * lin_space, False)
    minimum = find_min(kde_pdf, color_max, background_max)

    # complete correct reindexing
    color_max = color_max / lin_space + minimum_grey
    background_max = background_max / lin_space + minimum_grey
    minimum = minimum / lin_space + minimum_grey

    # calculate the cutoff value for greys
    cutoff_value = (background_max + minimum) / 2

    # add column to dataframe that classifies each image as background or not
    df['background'] = df.apply(lambda row: is_it_background(cutoff_value, row['greys']), axis=1)

    return df


def main_preprocessing(complete_path, training=True, save_im=True, max_tile_x=4000, max_tile_y=3000):
    """
    The primary function to perform all preprocessing
    of the data, creating patches and returning a final
    large dataframe with all information contained.

    Parameters
    -----
    complete_path: the full path to the file containing all svs
        files that will be used for training the model or a single
        svs file to get an output value
    training: a boolean that indicates if this preprocessing is
        for training data or if it to only be used for the existing model
    save_im: a boolean that indicates if tissue images should be saved 
        (beware this is a lot of data, at least 10GB per slide)
    max_tile_x: the maximum x dimension size, in pixels,
        of a slide patch (default is 4000)
    max_tile_y: the maximum y dimension size, in pixels,
        of a slide patch (default is 3000)

    Returns
    -----
    all_df or sorted_df: a dataframe containing all necessary information for
        creating superpatches for training (all_df) or for inputting into an
        already generated model (sorted_df)
    """

    if training:

        all_df = pd.DataFrame()

        # iterate through all files in the directory
        for file in os.listdir(complete_path):

            # check that the file is a slide image
            if file.endswith('.svs'):

                # open the slide file
                slide_file, slide_x, slide_y = open_slide(file)

                # calculate dimensions and losses
                ydim, ypatch, yloss = get_tile_size(max_tile_y, slide_y)
                xdim, xpatch, xloss = get_tile_size(max_tile_x, slide_x)
                loss_percentage = percent_of_pixels_lost(xloss, xpatch, 
                                                        yloss, ypatch, 
                                                        slide_x, slide_y)
                
                # get the dataframe for all patches in the slide
                dataframe_patches = compile_patch_data(slide_file, ypatch, 
                                                    xpatch, xdim, ydim)

                # determine if patches are background or not
                sorted_df = sort_patches(dataframe_patches)

                # drop all background images from dataframe
                sorted_df = sorted_df.loc[~sorted_df.background, :]

                # save all images to correct directory if desired
                if save_im: save_all_images(sorted_df, complete_path, file)

                # create a unique id for this slide image
                sorted_df['UUID'] = uuid.uuid4()

                # add the dataframe to the total training dataframe
                all_df = pd.concat([all_df, sorted_df], ignore_index=True)

                # print out the percent of pixels lost
                print(f'Percent of pixels lost in pre-processing for {file}: {loss_percentage} %')

            # if the file is not a slide image then do nothing and continue
            else:
                continue

        # give unique numeric ID to each slide counting from 0 upwards
        all_df['id'] = pd.factorize(all_df['UUID'])[0]

        # remove UUID column across the entire dataframe
        all_df = all_df.drop(columns='UUID')
        
        return all_df

    else:
        
        # open the slide file
        slide_file, slide_x, slide_y = open_slide(complete_path)

        # calculate dimensions and losses
        ydim, ypatch, yloss = get_tile_size(max_tile_y, slide_y)
        xdim, xpatch, xloss = get_tile_size(max_tile_x, slide_x)
        loss_percentage = percent_of_pixels_lost(xloss, xpatch, 
                                                yloss, ypatch, 
                                                slide_x, slide_y)
        
        # get the dataframe for all patches in the slide
        dataframe_patches = compile_patch_data(slide_file, ypatch, 
                                            xpatch, xdim, ydim)

        # determine if patches are background or not
        sorted_df = sort_patches(dataframe_patches)

        # drop all background images from dataframe
        sorted_df = sorted_df.loc[~sorted_df.background, :]

        # save all images to correct directory if desired
        if save_im: save_all_images(sorted_df, complete_path, file)

        # print out the percent of pixels lost
        print(f'Percent of Pixels Lost in Pre-Processing: {loss_percentage} %')

        return sorted_df


def count_images():
    """
    Count images finds the number of whole slide images available
    in your current working directory. 

    Parameters
    ------------
    None

    Returns:
    -----------
    img_count (int): the number of whole slide images in your directory
    """
    
    cwd = os.getcwd()
    file_list = os.listdir(cwd)
    
    # count the number of svs images in cwd
    img_count = 0
    for file in file_list:
        if file.endswith('.svs'):
            img_count += 1
        else:
            continue
        
    return img_count


def patches_per_img(num_patches):

    # find the number of images in cwd
    img_count = count_images()

    # find the patches per image
    patch_img  = num_patches/img_count

    # return patches per image
    return patch_img


def get_superpatch_patches(patches_df, patches=6):
   """
   This function finds the patches to comprise the superpatch.
   The patches are selected based off of distribution of 
   average color and the source image. This way, the superpatch
   is not entirely made of patches from one image (unless there is
   only one image available).

   Parameters:
   -------------
   df (pandas df): MUST be dataframe from main_preprocessing output

   Returns:
   -------------
   patches_list: list of the patches to be included in superpatch
                 individual patches are stored as np arrays
   """

   # remove all unnecessary columns
   df = patches_df.drop(['background', 'RGB_avg'], axis=1)

   # make sure there are enough patches
   if len(df.index) >= patches:
      pass
   else:
    raise IndexError('Fewer patches available in dataframe than requested for superpatch')
   
   # force number of patches to be even
   if patches % 2 == 0:
      pass
   else:
      ValueError('Number of patches must be an even integer')
   
   # patches list
   patches_list = []

   # calculate patches per image
   # need to change this
   patch_per = math.floor(patches_per_img(patches))
   
   # bin the average values for each patch
   df['grey_binned'] = pd.cut(df['greys'], bins=(patches+1))

   # find the bins and the img_labels
   bins = df['grey_binned'].unique()
   img_labels = df['id'].unique()

   for img in img_labels:

      # get only the patches of the image
      img_df = df.loc[df['id'] == img]

      # start counting the number of patches
      # from this image
      patch_count = 0

      for bin_i in bins:

         # get the dataframe that contains the
         # patches with the average color of interest
         bin_df = img_df.loc[img_df['grey_binned'] == bin_i]

         # pick a patch from this set of relevant patches
         patch_df = bin_df.sample()
         actual_patch = patch_df['patches']

         # add patch to list of patches to access later
         patches_list.append(actual_patch)

         # count the patches used for this image
         patch_count += 1
         
         # get only the number of patches per image
         if patch_count >= patch_per:
            # leave the for loop if you have the number of patches
            break
         else:
            # keep looping if you need more patches from this image
            continue
   
   # return the list of patches that make up the superpatch
   return patches_list


def superpatcher(patches_list, sp_width=3):
    """
    Superpatcher uses the selected patches and
    converts the individual patches into one patch

    Parameters:
    ------------
    patches_list: MUST be output from get_superpatch_patches
                  list of patches
    sp_width: the width of a superpatch (how many images, default 3)

    Returns:
    --------------
    superpatch: np.array that contains the superpatch
    """

    num_patches = len(patches_list)
    sp_height_calc = math.ceil(num_patches/sp_width)
    sp_width_calc = int(num_patches/sp_height_calc)

    # initialize the row patch (starting column, adding column wise)
    patch_array_0 = (patches_list[0]).values[0]
    patch_index = 2

    for j in range(0, sp_height_calc):
        
        # build the row (build the row)
        for i in range(1, sp_width_calc):

            # get patch at index i (in list)
            patch_array_i = (patches_list[patch_index]).values[0]
            
            patch_index += 1
            
            # update the overall patch to be these patched together
            patch_array_0 = np.concatenate((patch_array_0, patch_array_i), axis=1)

            # save the finished row
            if i == (sp_width_calc-1):
                patch_row_0 = patch_array_0
                patch_array_0 = (patches_list[j+1]).values[0]

        # find the first row (adding row wise)
        if j == 0:
            patch_row_1 = patch_row_0

        # else add the row to the other rows
        else:
            patch_row_1 = np.concatenate((patch_row_0, patch_row_1), axis=0)
        
    return patch_row_1


def preprocess(path, patches=6, training=True, save_im=True, max_tile_x=4000, max_tile_y=3000):
    if training:
        dataframe = main_preprocessing(path, training, save_im, max_tile_x, max_tile_y)
        plist = get_superpatch_patches(dataframe, patches)
        spatch = superpatcher(plist)
        save_image(path, 'superpatch_training.tif', spatch)

    else:
        main_preprocessing(path, training, save_im, max_tile_x, max_tile_y)

    return spatch