import os
import numpy as np
import SimpleITK as sitk

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path, sub_directories and all_sub_files
    :param file_dir:
    :return: dirs or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            return dirs
        if len(files) and file:
            return files

def norm_intens_filter(slice, top=99, bottom=1):
    """
    remove the top and bottom 1% of intensities and then normalise image with zero mean and unit variance  
    :param slice:
    :param top:
    :param bottom:
    """
    # removing outlier intensities
    b = np.percentile(slice, top)
    t = np.percentile(slice, bottom)
    slice = np.clip(slice, t, b)

    # the non-background regions should be normalised since intensities range from 0 to 5000
    # the min in the normalised slice corresponds to 0 intensity in unnormalised slice
    # this is replaced with -9 to keep track of 0 intensities
    # so those intensities can be discarded after during random patch sampling
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        norm = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        norm[norm == norm.min()] = -9 # black background area
        return norm

def crop_center(img, croph, cropw):
    """
    crop slices from their center by a certain height and width.  
    :param img:
    :param croph:
    :param cropw:
    """   
    height, width = img[0].shape 
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)        
    return img[:, starth: starth+croph, startw: startw + cropw]


