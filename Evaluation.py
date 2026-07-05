#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Evaluation of LFW CFP-FP and AgeDB-30 data

@author: AIRocker
"""
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from face_model import MobileFaceNet, l2_norm
from data_set.dataloader import LFW, CFP_FP, AgeDB30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getAccuracy(scores, flags, threshold, method):
    if method == 'l2_distance':
        p = np.sum(scores[flags == 1] < threshold)
        n = np.sum(scores[flags == -1] > threshold)
    elif method == 'cos_distance':
        p = np.sum(scores[flags == 1] > threshold)
        n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum, method):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 3.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i], method)
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def getFeature(net, dataloader, device, flip = True):
### Calculate the features ###
    featureLs = None
    featureRs = None 
    count = 0
    for det in dataloader:
        for i in range(len(det)):
            det[i] = det[i].to(device)
        count += det[0].size(0)
#        print('extracing deep features from the face pair {}...'.format(count))
    
        with torch.no_grad():
            res = [net(d).data.cpu() for d in det]
            
        if flip:      
            featureL = l2_norm(res[0] + res[1])
            featureR = l2_norm(res[2] + res[3])
        else:
            featureL = res[0]
            featureR = res[2]
        
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = torch.cat((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = torch.cat((featureRs, featureR), 0)
        
    return featureLs, featureRs

def evaluation_10_fold(featureL, featureR, dataset, method = 'l2_distance'):
    """
    10-fold cross-validation for face verification.
    Each fold must use a fresh copy of features so normalization (mean subtract + L2 norm)
    is computed from that fold's validation set only; mutating in-place would corrupt later folds.
    """
    ACCs = np.zeros(10)
    threshold = np.zeros(10)
    fold = np.array(dataset.folds).reshape(1, -1)
    flags = np.array(dataset.flags).reshape(1, -1)
    flags_1d = np.squeeze(flags)
    # Keep originals; copy per fold so we don't mutate across folds
    featureL_np = featureL.numpy() if hasattr(featureL, 'numpy') else np.asarray(featureL)
    featureR_np = featureR.numpy() if hasattr(featureR, 'numpy') else np.asarray(featureR)

    for i in range(10):
        valFold = (fold != i).ravel()
        testFold = (fold == i).ravel()

        # Copy so this fold's normalization doesn't affect others
        featureLs = featureL_np.copy()
        featureRs = featureR_np.copy()

        mu = np.mean(np.concatenate((featureLs[valFold, :], featureRs[valFold, :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        if method == 'l2_distance':
            scores = np.sum(np.power((featureLs - featureRs), 2), 1)
        elif method == 'cos_distance':
            scores = np.sum(np.multiply(featureLs, featureRs), 1)

        threshold[i] = getThreshold(scores[valFold], flags_1d[valFold], 10000, method)
        ACCs[i] = getAccuracy(scores[testFold], flags_1d[testFold], threshold[i], method)

    return ACCs, threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face_Detection_Evaluation')
    parser.add_argument('--dataset', type=str, default='LFW', help='Select the dataset to evaluate, LFW, CFP-FP, AgeDB-30')
    parser.add_argument('--method', type=str, default='l2_distance', 
                        help='methold to calculate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    args = parser.parse_args()
    
    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')

    detect_model.eval()

    ### load data ###
    transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    select_dataset = args.dataset
    
    if select_dataset == 'LFW':
    
        root = 'data_set/LFW/lfw_align_112'
        file_list = 'data_set/LFW/pairs.txt'
        dataset = LFW(root, file_list, transform=transform)
        
    elif select_dataset == 'CFP-FP':
    
        root = 'data_set/CFP-FP/CFP_FP_aligned_112'
        file_list = 'data_set/CFP-FP/cfp_fp_pair.txt'
        dataset = CFP_FP(root, file_list, transform=transform)
        
    elif select_dataset == 'AgeDB-30':
        
        root = 'data_set/AgeDB-30/agedb30_align_112'
        file_list = 'data_set/AgeDB-30/agedb_30_pair.txt'
        dataset = AgeDB30(root, file_list, transform=transform)    
    
    dataloader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    print('{} data is loaded with length'.format(select_dataset), len(dataset))
    
    featureLs, featureRs = getFeature(detect_model, dataloader, device, flip = args.flip)
    scores_l2 = np.sum(np.power((featureLs.numpy() - featureRs.numpy()), 2), 1) # L2 distance
    
    ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset, method = args.method)
    
    for i in range(len(ACCs)):
        print('{} accuracy: {:.2f} threshold: {:.4f}'.format(i+1, ACCs[i] * 100, threshold[i]))
    print('--------')
    print('Average Acc:{:.4f} Average Threshold:{:.4f}'.format(np.mean(ACCs) * 100, np.mean(threshold)))