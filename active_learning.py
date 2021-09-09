from glob import glob
import os
import errno
import sys
import time
from collections import OrderedDict
import imageio
import logging

import numpy as np
from skimage.io import imread, imsave 
from hausdorff import hausdorff_distance
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as tf
import torch.utils.data as data
from torch.cuda import amp
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import pickle
import pandas as pd

from train_val import train, validate
from loss import BCEDiceLoss
from metrics import *
from utils import *
from model import *
from dataset import *

def enable_dropout(model):
    """
    enable the dropout layers during test-time 
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_samples(data_loader, forward_passes, iteration, nb_active_learning_iter_size, model, n_classes, n_samples, device, height, width):
    """
    a pool of unlabelled images is fed into the trained U-Net and a measure of uncertainty is computed for each unlabelled sample. 
    entropy is used as a measure of uncertainty and is taken across different predictions.  
    :param data_loader:
    :param forward_passes:
        number of monte-carlo samples/forward passes
    :param model:
        unet
    :param n_classes:
        number of classes in the dataset
    :param n_samples:
        number of samples in the test set
    :param device:
    """
    dropout_predictions = np.empty((0, n_samples-(iteration*nb_active_learning_iter_size), n_classes, height, width), dtype=np.float32) 
    for i in tqdm(range(forward_passes)):                         # perform x forward passes to obtain x monte-carlo predictions
        predictions_forward_pass = np.empty((0, n_classes, height, width))
        model.eval()
        enable_dropout(model)   
        with torch.no_grad():            
            for batch_idx, (data, labels) in enumerate(data_loader):
                data = data.to(device) 
                preds = model(data)
                preds = torch.sigmoid(preds).data.cpu().numpy()
                predictions_forward_pass = np.vstack((predictions_forward_pass, preds))   
        dropout_predictions = np.vstack((dropout_predictions, predictions_forward_pass[np.newaxis, :, :, :, :])) 
    mean = np.mean(dropout_predictions, axis=0) 
    epsilon = sys.float_info.min
    uncertainty_maps = -np.sum(mean*np.log(mean + epsilon), axis=1)
    avg_pred = np.mean(dropout_predictions, axis=0)
    dropout_predictions = 0
    predictions_forward_pass = 0
    return avg_pred, uncertainty_maps

def uncertainty_scoring(uncertainty_maps):
    """
    uncertainty maps are given an uncertainty score 
    :param uncertainty_maps: trained Unet model
    :return: numpy array with an avg uncertainty score for each image in the unlabelled set
    """
    uncertainty_scores = np.empty((uncertainty_maps.shape[0]))
    for i in range(uncertainty_maps.shape[0]):
        uncertainty_scores[i] = np.mean(uncertainty_maps[i])    
    overall_uncertainty_score = np.sum(uncertainty_maps)
    return uncertainty_scores, overall_uncertainty_score
    
def add_annotated_sample(to_be_annotated_index, labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths):
    """
    the labelled dataset is amended to include the samples to be annotated and the unlabelled dataset is amended to remove the samples to be annotated.
    :param labelled_dataset:
    :param to_be_annotated_index:
    :param labelled_img_paths:
    :param labelled_mask_paths:
    :param unlabelled_img_paths:
    :param unlabelled_mask_paths:
    :param batch_size:
    """
    samples_to_be_annotated = [unlabelled_img_paths[i] for i in to_be_annotated_index]
    annotation = [unlabelled_mask_paths[i] for i in to_be_annotated_index]
    new_labelled_img_paths = labelled_img_paths + samples_to_be_annotated
    new_labelled_mask_paths = labelled_mask_paths + annotation 
    new_unlabelled_img_paths = list(np.delete(unlabelled_img_paths, to_be_annotated_index))
    new_unlabelled_mask_paths = list(np.delete(unlabelled_mask_paths, to_be_annotated_index))
    return new_labelled_img_paths, new_labelled_mask_paths, new_unlabelled_img_paths, new_unlabelled_mask_paths

def main():
    TRAIN_IMG_DIR = pickle.load(open('data\\train_val\\img\\'+'train_val.data', 'rb'))
    TRAIN_LABEL_DIR = pickle.load(open('data\\train_val\\label\\'+'train_val.mask', 'rb'))
    UNLABELLED_IMG_DIR = pickle.load(open('data\\unlabelled\\img\\'+'unlabelled.data', 'rb'))
    UNLABELLED_LABEL_DIR = pickle.load(open('data\\unlabelled\\label\\'+'unlabelled.mask', 'rb'))
    TEST_IMG_DIR = pickle.load(open('data\\test\\img\\'+'test.data', 'rb'))
    TEST_LABEL_DIR = pickle.load(open('data\\test\\label\\'+'test.mask', 'rb'))

    # split data into train/val (labelled, 20%), unlabelled (active learning simulation, 60%), test (20%) - using 5000 slices
    random_bool = False                       # controls whether the labelled set is entered randomly - normal training or actively - active learning 
    height, width = 160, 160
    n_initial_unlabelled_samples = len(UNLABELLED_IMG_DIR)
    n_classes = 3                             # edema, non-enhancing tumour, enhancing tumour
    nb_experiments = 3                        # number of experiments
    nb_active_learning_iter = 15              # number of active learning iterations 15
    nb_active_learning_iter_size = 30         # number of samples to be added to the training set after each active learning iteration - number of labels requested from oracle 20
    FORWARD_PASSES = 10                       # number of monte carlo predictions are used to calculate uncertainty 15
    EPOCHS = 200
    LEARNING_RATE = 1e-3
    EARLY_STOP = 25                           # 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32

    if random_bool == True:
        learning_type = 'random'
    else:
        learning_type = 'active'
    logging.basicConfig(level=logging.INFO,                                         # instantiate a logger
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= learning_type + '_learning_results/'+ learning_type + '_learning_output.txt',
                    filemode='w')
    
    console = logging.StreamHandler()                                               # define a new Handler to log to console as well
    console.setLevel(logging.INFO)                                                  # set the logging level
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')      # set a format which is the same for console use
    console.setFormatter(formatter)                                                 # tell the handler to use this format
    logging.getLogger('').addHandler(console)                                       # add the handler to the root logger

    # model, optimiser
    model = UNet2D(in_channels=4, out_channels=3).to(DEVICE) 
    logging.info("=> Creating 2D UNET Model")
    loss_fn = BCEDiceLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # initisialise results trackers
    mean_WT_dice = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_TC_dice = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_ET_dice = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_WT_PPV = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_TC_PPV = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_ET_PPV = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_WT_sensitivity = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_TC_sensitivity = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_ET_sensitivity = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_WT_Hausdorff = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_TC_Hausdorff = np.zeros((nb_experiments, nb_active_learning_iter))
    mean_ET_Hausdorff = np.zeros((nb_experiments, nb_active_learning_iter))
    std_WT_dice = np.zeros((nb_experiments, nb_active_learning_iter))
    std_TC_dice = np.zeros((nb_experiments, nb_active_learning_iter))
    std_ET_dice = np.zeros((nb_experiments, nb_active_learning_iter))
    std_WT_PPV = np.zeros((nb_experiments, nb_active_learning_iter))
    std_TC_PPV = np.zeros((nb_experiments, nb_active_learning_iter))
    std_ET_PPV = np.zeros((nb_experiments, nb_active_learning_iter))
    std_WT_sensitivity = np.zeros((nb_experiments, nb_active_learning_iter))
    std_TC_sensitivity = np.zeros((nb_experiments, nb_active_learning_iter))
    std_ET_sensitivity = np.zeros((nb_experiments, nb_active_learning_iter))
    std_WT_Hausdorff = np.zeros((nb_experiments, nb_active_learning_iter))
    std_TC_Hausdorff = np.zeros((nb_experiments, nb_active_learning_iter))
    std_ET_Hausdorff = np.zeros((nb_experiments, nb_active_learning_iter))

    overall_start = time.time()
    for r in range(nb_experiments):        
        logging.info("\n*****************EXPERIMENT " + str(r+1) + " IS STARTING********************")
        
        # initialise paths with original unlabelled and labelled datasets 
        labelled_img_paths, labelled_mask_paths = TRAIN_IMG_DIR, TRAIN_LABEL_DIR
        unlabelled_img_paths, unlabelled_mask_paths = UNLABELLED_IMG_DIR, UNLABELLED_LABEL_DIR
        test_img_paths, test_mask_paths = TEST_IMG_DIR, TEST_LABEL_DIR
        
        # keep the validation set the same for each experiment, hence each active learning iteration
        labelled_img_paths, val_img_paths, labelled_mask_paths, val_mask_paths = train_test_split(labelled_img_paths, 
                                            labelled_mask_paths, test_size=0.2, random_state=41)

        for i in range(nb_active_learning_iter):                    # how many times the active learning loops
            # # reset the weights
            # model.load_state_dict(torch.load('models/base_trained/2DUNET.pth'))
            if random_bool == True:
                model_path = 'models/random_trained/2DUNET_experiment_' + str(r+1) + '_iter_' + str(i) + '.pth'
                results_path = 'random_learning_results'
                data_type = 'random'
            else:
                model_path = 'models/active_trained/2DUNET_experiment_' + str(r+1) + '_iter_' + str(i) + '.pth'
                results_path = 'active_learning_results'
                data_type = 'uncertain'
            
            if i == 0:
                # load weights from base training set
                model.load_state_dict(torch.load('models/base_trained/2DUNET.pth'))
            
            path = results_path + "/experiment_" + str(r+1) + "/AL_iteration_" + str(i)
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            logging.info("\n---------STARTING AL ITERATION NUMBER " + str(i) + "----------")

            # load unlabelled dataset
            unlabelled_dataset = Dataset(unlabelled_img_paths, unlabelled_mask_paths)
            unlabelled_loader = torch.utils.data.DataLoader(
                unlabelled_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False)

            # select samples to be annotated by an expert, either randomly or based on an uncertainty measure
            if random_bool == True:
                to_be_added_random = np.random.choice(len(unlabelled_img_paths), nb_active_learning_iter_size)
                labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, \
                    unlabelled_mask_paths = add_annotated_sample(to_be_added_random, labelled_img_paths, 
                                                        labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths)
            else:
                logging.info("Computing Monte Carlo predictions for unlabelled data ...\n")
                pred, uncertainty_maps = mc_samples(unlabelled_loader, FORWARD_PASSES, i, nb_active_learning_iter_size, # pred = 1500, 3, 160, 160 , uncertainty_maps = 1500, 160, 160
                                    model, n_classes, n_initial_unlabelled_samples, DEVICE, height, width)
                if r == 0:
                    np.save(path + '/avg_predictions.npy', pred[:1000,:,:,:])
                    np.save(path + '/uncertainty_maps.npy', uncertainty_maps[:1000,:,:])
                uncertainty_scores, overall_uncertainty_score = uncertainty_scoring(uncertainty_maps)
                to_be_annotated_uncertain = uncertainty_scores.argsort()[::-1][:nb_active_learning_iter_size]       # sorting the samples with the top x uncertainties and indexing them          
                labelled_img_paths, labelled_mask_paths, unlabelled_img_paths, \
                    unlabelled_mask_paths = add_annotated_sample(to_be_annotated_uncertain, labelled_img_paths, 
                                                        labelled_mask_paths, unlabelled_img_paths, unlabelled_mask_paths)
                np.save('active_learning_results\\labelled_img_paths.npy', labelled_img_paths)                       
                np.save('active_learning_results\\labelled_mask_paths.npy', labelled_mask_paths)                       
                np.save('active_learning_results\\unlabelled_img_paths.npy', unlabelled_img_paths)                       
                np.save('active_learning_results\\unlabelled_mask_paths.npy', unlabelled_mask_paths)                       
                pred, uncertainty_maps = 0, 0
            
            if random_bool == False:
                logging.info("\n Overall uncertainty score for active learning iteration " + str(i) + " is " + str(overall_uncertainty_score))
                logging.info("\n Sample indexes " + str(to_be_annotated_uncertain) + " have been classified with high uncertainty")
            
            logging.info(str(nb_active_learning_iter_size) + " " + data_type + " samples have been added to the training set \n")

            logging.info("Training set now contains " + str(len(labelled_img_paths)) + " samples \n")

            logging.info("------------TRAINING FOR AL ITERATION NUMBER " + str(i) + "-----------\n")

            train_dataset = Dataset(labelled_img_paths, labelled_mask_paths)
            val_dataset = Dataset(val_img_paths, val_mask_paths)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False)

            log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'
            ])

            best_dice = 0
            trigger = 0
            start = time.time()
            for epoch in range(EPOCHS):
                logging.info("'Epoch [%d/%d]" %(epoch, EPOCHS))

                # train
                train_log = train(train_loader, model, loss_fn, optimizer, DEVICE)

                # validate
                val_log = validate(val_loader, model, loss_fn, DEVICE)

                logging.info('loss %.4f - iou %.4f - dice - %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
                    %(train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))
                
                tmp = pd.Series([
                    epoch,
                    LEARNING_RATE,
                    train_log['loss'],
                    train_log['iou'],
                    train_log['dice'],
                    val_log['loss'],
                    val_log['iou'],
                    val_log['dice'],
                ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'])

                log = log.append(tmp, ignore_index=True)
                
                log.to_csv(path + "/log" + ".csv", index=False)

                trigger += 1

                if val_log['dice'] > best_dice:
                    torch.save(model.state_dict(), model_path)
                    best_dice = val_log['dice']
                    logging.info("=> best model has been saved")
                    trigger = 0

                # early stopping
                if trigger >= EARLY_STOP:
                    logging.info("=> early stopping")
                    break
                torch.cuda.empty_cache()
            end = time.time()
            logging.info('Training and validation for active learning iteration ' + str(i) + ' has taken ' + str((end - start)/60) + ' minutes to complete')
    
            
            logging.info("------------TESTING AFTER ACTIVE LEARNING ITERATION " + str(i) + " -----------\n")

            test_dataset = Dataset(test_img_paths, test_mask_paths)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                drop_last=False)
            
            model.load_state_dict(torch.load(model_path))
            model.eval()
            """
            obtain and save the label map generated by the model for each active learning iteration
            """
            with torch.no_grad():
                for batch_idx, (data, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    data = data.to(DEVICE) 
                    
                    preds = model(data)
                    preds = torch.sigmoid(preds).data.cpu().numpy()
                    
                    img_paths = test_img_paths[BATCH_SIZE*batch_idx:BATCH_SIZE*(batch_idx+1)]

                    for b in range(preds.shape[0]):
                        npName = os.path.basename(img_paths[b])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(preds.shape[2]):
                            for idy in range(preds.shape[3]):
                                #(ED, peritumoral edema) (label 2) green
                                if preds[b,0,idx,idy] > 0.5: 
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                #(NET, non-enhancing tumor)(label 1) red
                                if preds[b,1,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                #(ET, enhancing tumor)(label 4) yellow
                                if preds[b,2,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0

                        test_pred_path = path + "/output/"
                        try:
                            os.makedirs(test_pred_path)
                        except OSError as exc:
                            if exc.errno != errno.EEXIST:
                                raise
                            pass
                        imsave(test_pred_path + rgbName, rgbPic, check_contrast=False)

            """
            calculate metrics: Dice, Sensitivity, PPV, Hausdorff for each AL iteration
            """
            wt_dices = []
            tc_dices = []
            et_dices = []
            wt_sensitivities = []
            tc_sensitivities = []
            et_sensitivities = []
            wt_ppvs = []
            tc_ppvs = []
            et_ppvs = []
            wt_Hausdorff = []
            tc_Hausdorff = []
            et_Hausdorff = []

            maskPath = glob("base_test_results/output/" + "GT\*.png")
            pbPath = glob(path + "/output/" + "*.png")

            for myi in tqdm(range(len(maskPath))):
                mask = imread(maskPath[myi])
                pb = imread(pbPath[myi])

                wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
                wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

                tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
                tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

                etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
                etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

                for idx in range(mask.shape[0]):
                    for idy in range(mask.shape[1]):
                        # As long as any channel of this pixel has a value, it means that this pixel does not belong to the foreground, that is, it belongs to the WT area 
                        if mask[idx, idy, :].any() != 0:
                            wtmaskregion[idx, idy] = 1
                        if pb[idx, idy, :].any() != 0:
                            wtpbregion[idx, idy] = 1
                        # As long as the first channel is 255, it can be judged to be the TC area, because the first channel of red and yellow are both 255, which is different from green 
                        if mask[idx, idy, 0] == 255:
                            tcmaskregion[idx, idy] = 1
                        if pb[idx, idy, 0] == 255:
                            tcpbregion[idx, idy] = 1
                        # As long as the second channel is 128, it can be judged to be the ET area 
                        if mask[idx, idy, 1] == 128:
                            etmaskregion[idx, idy] = 1
                        if pb[idx, idy, 1] == 128:
                            etpbregion[idx, idy] = 1

                #Start calculating WT - whole tumour
                dice = dice_coef(wtpbregion,wtmaskregion)
                wt_dices.append(dice)
                ppv_n = ppv(wtpbregion, wtmaskregion)
                wt_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
                wt_Hausdorff.append(Hausdorff)
                sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
                wt_sensitivities.append(sensitivity_n)

                # Start calculating TC - tumour core
                dice = dice_coef(tcpbregion, tcmaskregion)
                tc_dices.append(dice)
                ppv_n = ppv(tcpbregion, tcmaskregion)
                tc_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
                tc_Hausdorff.append(Hausdorff)
                sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
                tc_sensitivities.append(sensitivity_n)

                # Start calculating ET - enhancing tumour
                dice = dice_coef(etpbregion, etmaskregion)
                et_dices.append(dice)
                ppv_n = ppv(etpbregion, etmaskregion)
                et_ppvs.append(ppv_n)
                Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
                et_Hausdorff.append(Hausdorff)
                sensitivity_n = sensitivity(etpbregion, etmaskregion)
                et_sensitivities.append(sensitivity_n)

            logging.info("------------RESULTS FOR EXPERIMENT " + str(r+1) + " AND ACTIVE LEARNING ITERATION " + str(i) + " -----------\n")

            logging.info('WT Dice: %.4f' % np.mean(wt_dices))
            logging.info('TC Dice: %.4f' % np.mean(tc_dices))
            logging.info('ET Dice: %.4f' % np.mean(et_dices))
            logging.info("=============")
            logging.info('WT PPV: %.4f' % np.mean(wt_ppvs))
            logging.info('TC PPV: %.4f' % np.mean(tc_ppvs))
            logging.info('ET PPV: %.4f' % np.mean(et_ppvs))
            logging.info("=============")
            logging.info('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
            logging.info('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
            logging.info('ET sensitivity: %.4f' % np.mean(et_sensitivities))
            logging.info("=============")
            logging.info('WT Hausdorff: %.4f' % np.mean(wt_Hausdorff))
            logging.info('TC Hausdorff: %.4f' % np.mean(tc_Hausdorff))
            logging.info('ET Hausdorff: %.4f' % np.mean(et_Hausdorff))
            logging.info("=============")

            mean_WT_dice[r][i] = np.mean(wt_dices) 
            mean_TC_dice[r][i] = np.mean(tc_dices) 
            mean_ET_dice[r][i] = np.mean(et_dices) 
            mean_WT_PPV[r][i] = np.mean(wt_ppvs) 
            mean_TC_PPV[r][i] = np.mean(tc_ppvs) 
            mean_ET_PPV[r][i] = np.mean(et_ppvs) 
            mean_WT_sensitivity[r][i] = np.mean(wt_sensitivities) 
            mean_TC_sensitivity[r][i] = np.mean(tc_sensitivities) 
            mean_ET_sensitivity[r][i] = np.mean(et_sensitivities) 
            mean_WT_Hausdorff[r][i] = np.mean(wt_Hausdorff) 
            mean_TC_Hausdorff[r][i] = np.mean(tc_Hausdorff) 
            mean_ET_Hausdorff[r][i] = np.mean(et_Hausdorff) 
            std_WT_dice[r][i] = np.std(wt_dices) 
            std_TC_dice[r][i] = np.std(tc_dices) 
            std_ET_dice[r][i] = np.std(et_dices) 
            std_WT_PPV[r][i] = np.std(wt_ppvs) 
            std_TC_PPV[r][i] = np.std(tc_ppvs) 
            std_ET_PPV[r][i] = np.std(et_ppvs) 
            std_WT_sensitivity[r][i] = np.std(wt_sensitivities) 
            std_TC_sensitivity[r][i] = np.std(tc_sensitivities) 
            std_ET_sensitivity[r][i] = np.std(et_sensitivities) 
            std_WT_Hausdorff[r][i] = np.std(wt_Hausdorff) 
            std_TC_Hausdorff[r][i] = np.std(tc_Hausdorff) 
            std_ET_Hausdorff[r][i] = np.std(et_Hausdorff) 

            np.save(results_path + '/mean_WT_dice.npy', mean_WT_dice) 
            np.save(results_path + '/mean_TC_dice.npy', mean_TC_dice)   
            np.save(results_path + '/mean_ET_dice.npy', mean_ET_dice) 
            np.save(results_path + '/mean_WT_PPV.npy', mean_WT_PPV) 
            np.save(results_path + '/mean_TC_PPV.npy', mean_TC_PPV) 
            np.save(results_path + '/mean_ET_PPV.npy', mean_ET_PPV) 
            np.save(results_path + '/mean_WT_sensitivity.npy', mean_WT_sensitivity) 
            np.save(results_path + '/mean_TC_sensitivity.npy', mean_TC_sensitivity)  
            np.save(results_path + '/mean_ET_sensitivity.npy', mean_ET_sensitivity) 
            np.save(results_path + '/mean_WT_Hausdorff.npy', mean_WT_Hausdorff) 
            np.save(results_path + '/mean_TC_Hausdorff.npy', mean_TC_Hausdorff)  
            np.save(results_path + '/mean_ET_Hausdorff.npy', mean_ET_Hausdorff)
            np.save(results_path + '/std_WT_dice.npy', std_WT_dice)  
            np.save(results_path + '/std_TC_dice.npy', std_TC_dice) 
            np.save(results_path + '/std_ET_dice.npy', std_ET_dice)  
            np.save(results_path + '/std_WT_PPV.npy', std_WT_PPV) 
            np.save(results_path + '/std_TC_PPV.npy', std_TC_PPV)  
            np.save(results_path + '/std_ET_PPV.npy', std_ET_PPV) 
            np.save(results_path + '/std_WT_sensitivity.npy', std_WT_sensitivity)  
            np.save(results_path + '/std_TC_sensitivity.npy', std_TC_sensitivity) 
            np.save(results_path + '/std_ET_sensitivity.npy', std_ET_sensitivity)  
            np.save(results_path + '/std_WT_Hausdorff.npy', std_WT_Hausdorff) 
            np.save(results_path + '/std_TC_Hausdorff.npy', std_TC_Hausdorff)  
            np.save(results_path + '/std_ET_Hausdorff.npy', std_ET_Hausdorff) 

    overall_end = time.time()     
    logging.info('Learning for ' + str(r+1) + ' experiments has taken ' + str((overall_end - overall_start)/60) + ' minutes to complete')

if __name__ == '__main__':
    main()