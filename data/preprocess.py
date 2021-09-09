## 2D BraTS Data Preprocessing

### Summary of steps performed:
# - N4ITK bias correction 
# - Normalise to zero mean and unit variance
# - 1% of the top and bottom intensities removed 
# - remove slices that have no tumour 

import os
import norm_intens_crop
import time
import numpy as np
import SimpleITK as sitk

flair_addon = "_flair.nii.gz"
t1_addon = "_t1.nii.gz"
t1ce_addon = "_t1ce.nii.gz"
t2_addon = "_t2.nii.gz"
label_addon = "_seg.nii.gz"
bratshgg_path = r"D:\\AI MSc Large Modules\\Masters_Project\\CODE\Deep-Inter-Active-Refinement-Network-for-Medical-Image-Segmentation\data\\MICCAI_BraTS_2018_Data_Training\\HGG"
bratslgg_path = r"D:\\AI MSc Large Modules\\Masters_Project\\CODE\Deep-Inter-Active-Refinement-Network-for-Medical-Image-Segmentation\\data\\MICCAI_BraTS_2018_Data_Training\\LGG"
outputImg_path = r".\data\\all_data\\img"
outputLabel_path = r".\\data\\all_data\\label"

pathhgg_list = norm_intens_crop.file_name_path(bratshgg_path)
pathlgg_list = norm_intens_crop.file_name_path(bratslgg_path)

# establish N4ITK bias field corrector object
corrector = sitk.N4BiasFieldCorrectionImageFilter()

start_HGG = time.time() 
# looping through each .nii.gz file containing HGGs
for subsetindex in range(len(pathhgg_list)):   
    brats_subset_path = bratshgg_path + "/" + str(pathhgg_list[subsetindex]) + "/"
    
    # get the four modalities and label of each folder
    flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_addon
    t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_addon
    t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_addon
    t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_addon
    label_image = brats_subset_path + str(pathhgg_list[subsetindex]) + label_addon
    
    # converting to sitk 
    flair_src = sitk.ReadImage(flair_image, sitk.sitkFloat32)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkFloat32)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkFloat32)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkFloat32)
    label = sitk.ReadImage(label_image, sitk.sitkUInt8)

    # performing N4ITK bias field correction on each MRI
    flair_n4itk_mask = sitk.OtsuThreshold(flair_src, 0, 1, 200)
    flair_n4itk = corrector.Execute(flair_src, flair_n4itk_mask)
    t1_n4itk_mask = sitk.OtsuThreshold(t1_src, 0, 1, 200)
    t1_n4itk = corrector.Execute(t1_src, t1_n4itk_mask)
    t1ce_n4itk_mask = sitk.OtsuThreshold(t1ce_src, 0, 1, 200)
    t1ce_n4itk = corrector.Execute(t1ce_src, t1ce_n4itk_mask)
    t2_n4itk_mask = sitk.OtsuThreshold(t2_src, 0, 1, 200)
    t2_n4itk = corrector.Execute(t2_src, t2_n4itk_mask)
    
    # getting arrays in sitk current size 240x240x155 each
    flair_array = sitk.GetArrayFromImage(flair_n4itk)
    t1_array = sitk.GetArrayFromImage(t1_n4itk)
    t1ce_array = sitk.GetArrayFromImage(t1ce_n4itk)
    t2_array = sitk.GetArrayFromImage(t2_n4itk)
    label_array = sitk.GetArrayFromImage(label)

    # normalise the four modalities separately since they have different contrasts
    flair_array_nor = norm_intens_crop.norm_intens_filter(flair_array)
    t1_array_nor = norm_intens_crop.norm_intens_filter(t1_array)
    t1ce_array_nor = norm_intens_crop.norm_intens_filter(t1ce_array)
    t2_array_nor = norm_intens_crop.norm_intens_filter(t2_array)

    # cropping each slice such that new size is 160x160x155 each
    flair_crop = norm_intens_crop.crop_center(flair_array_nor, 160, 160)
    t1_crop = norm_intens_crop.crop_center(t1_array_nor, 160, 160)
    t1ce_crop = norm_intens_crop.crop_center(t1ce_array_nor, 160, 160)
    t2_crop = norm_intens_crop.crop_center(t2_array_nor, 160, 160)
    label_crop = norm_intens_crop.crop_center(label_array, 160, 160) 

    # remove slices that do not contain tumour
    for n_slice in range(flair_crop.shape[0]):
        if np.max(label_crop[n_slice,:,:]) != 0:
            labelImg = label_crop[n_slice,:,:]
            
            four_modality_array = np.zeros((flair_crop.shape[1],flair_crop.shape[2],4), np.float)
            flairImg = flair_crop[n_slice,:,:]
            flairImg = flairImg.astype(np.float)
            four_modality_array[:,:,0] = flairImg

            t1Img = t1_crop[n_slice,:,:]
            t1Img = t1Img.astype(np.float)
            four_modality_array[:,:,1] = t1Img

            t1ceImg = t1ce_crop[n_slice,:,:]
            t1ceImg = t1ceImg.astype(np.float)
            four_modality_array[:,:,2] = t1ceImg

            t2Img = t2_crop[n_slice,:,:]
            t2Img = t2Img.astype(np.float)
            four_modality_array[:,:,3] = t2Img       
        
            imagepath = outputImg_path + "\\" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            labelpath = outputLabel_path + "\\" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(imagepath, four_modality_array) 
            np.save(labelpath, labelImg)
    
end_HGG = time.time()
print("HGG MRIs Preprocessed")
HGG_time = end_HGG-start_HGG
print('time taken to preprocess HGG MRIs = {} seconds'.format(HGG_time))

# repeat above process for LGG
start_LGG = time.time()
for subsetindex in range(len(pathlgg_list)):
    brats_subset_path = bratslgg_path + "/" + str(pathlgg_list[subsetindex]) + "/"

    flair_image = brats_subset_path + str(pathlgg_list[subsetindex]) + flair_addon
    t1_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1_addon
    t1ce_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t1ce_addon
    t2_image = brats_subset_path + str(pathlgg_list[subsetindex]) + t2_addon
    label_image = brats_subset_path + str(pathlgg_list[subsetindex]) + label_addon

    flair_src = sitk.ReadImage(flair_image, sitk.sitkFloat32)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkFloat32)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkFloat32)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkFloat32)
    label = sitk.ReadImage(label_image, sitk.sitkUInt8)

    flair_n4itk_mask = sitk.OtsuThreshold(flair_src, 0, 1, 200)
    flair_n4itk = corrector.Execute(flair_src, flair_n4itk_mask)
    t1_n4itk_mask = sitk.OtsuThreshold(t1_src, 0, 1, 200)
    t1_n4itk = corrector.Execute(t1_src, t1_n4itk_mask)
    t1ce_n4itk_mask = sitk.OtsuThreshold(t1ce_src, 0, 1, 200)
    t1ce_n4itk = corrector.Execute(t1ce_src, t1ce_n4itk_mask)
    t2_n4itk_mask = sitk.OtsuThreshold(t2_src, 0, 1, 200)
    t2_n4itk = corrector.Execute(t2_src, t2_n4itk_mask)
    
    flair_array = sitk.GetArrayFromImage(flair_n4itk)
    t1_array = sitk.GetArrayFromImage(t1_n4itk)
    t1ce_array = sitk.GetArrayFromImage(t1ce_n4itk)
    t2_array = sitk.GetArrayFromImage(t2_n4itk)
    label_array = sitk.GetArrayFromImage(label)

    flair_array_nor = norm_intens_crop.norm_intens_filter(flair_array)
    t1_array_nor = norm_intens_crop.norm_intens_filter(t1_array)
    t1ce_array_nor = norm_intens_crop.norm_intens_filter(t1ce_array)
    t2_array_nor = norm_intens_crop.norm_intens_filter(t2_array)

    flair_crop = norm_intens_crop.crop_center(flair_array_nor, 160, 160)
    t1_crop = norm_intens_crop.crop_center(t1_array_nor, 160, 160)
    t1ce_crop = norm_intens_crop.crop_center(t1ce_array_nor, 160, 160)
    t2_crop = norm_intens_crop.crop_center(t2_array_nor, 160, 160)
    label_crop = norm_intens_crop.crop_center(label_array, 160, 160) 

    for n_slice in range(flair_crop.shape[0]):
        if np.max(label_crop[n_slice,:,:]) != 0:
            labelImg = label_crop[n_slice,:,:]
            
            four_modality_array = np.zeros((flair_crop.shape[1],flair_crop.shape[2],4), np.float)
            flairImg = flair_crop[n_slice,:,:]
            flairImg = flairImg.astype(np.float)
            four_modality_array[:,:,0] = flairImg

            t1Img = t1_crop[n_slice,:,:]
            t1Img = t1Img.astype(np.float)
            four_modality_array[:,:,1] = t1Img

            t1ceImg = t1ce_crop[n_slice,:,:]
            t1ceImg = t1ceImg.astype(np.float)
            four_modality_array[:,:,2] = t1ceImg

            t2Img = t2_crop[n_slice,:,:]
            t2Img = t2Img.astype(np.float)
            four_modality_array[:,:,3] = t2Img       
        
            imagepath = outputImg_path + "\\" + str(pathlgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            labelpath = outputLabel_path + "\\" + str(pathlgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(imagepath, four_modality_array)
            np.save(labelpath, labelImg)

end_LGG = time.time()
print("LGG MRIs Preprocessed")
LGG_time = end_LGG-start_LGG
print('time taken to preprocess LGG MRIs = {} seconds'.format(LGG_time))