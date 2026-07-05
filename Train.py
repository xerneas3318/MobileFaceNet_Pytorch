#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:57:15 2019
Train Mobilefacenet

@author: AIRocker
"""
import os
import sys
import math
sys.path.append('..')
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from data_set.dataloader import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M
from face_model import MobileFaceNet, Arcface
import time
from Evaluation import getFeature, evaluation_10_fold

# Training data: prefer ~/Datasets/<name>, else data_set/
DATASETS_DIR = os.path.expanduser('~/Datasets')


def load_data(batch_size, dataset='Faces_emore', num_workers=4, pin_memory=True, prefetch_factor=4, use_dali=False, device_id=0):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]

    # Eval datasets: prefer ~/Datasets/<name>, else data_set/
    def _eval_root(name, subdir, pair_file):
        for base in (os.path.join(DATASETS_DIR, name), 'data_set/%s' % name):
            root = os.path.join(base, subdir)
            fl = os.path.join(base, pair_file)
            if os.path.isdir(root) and os.path.isfile(fl):
                return root, fl
        return os.path.join('data_set', name, subdir), os.path.join('data_set', name, pair_file)

    root, file_list = _eval_root('LFW', 'lfw_align_112', 'pairs.txt')
    dataset_LFW = LFW(root, file_list, transform=transform)
    print('Eval LFW: root={!r}, pairs={!r}, len={}'.format(os.path.abspath(root), os.path.abspath(file_list), len(dataset_LFW)))

    root, file_list = _eval_root('CFP-FP', 'CFP_FP_aligned_112', 'cfp_fp_pair.txt')
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
    print('Eval CFP-FP: root={!r}, pairs={!r}, len={}'.format(os.path.abspath(root), os.path.abspath(file_list), len(dataset_CFP_FP)))

    root, file_list = _eval_root('AgeDB-30', 'agedb30_align_112', 'agedb_30_pair.txt')
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)
    print('Eval AgeDB-30: root={!r}, pairs={!r}, len={}'.format(os.path.abspath(root), os.path.abspath(file_list), len(dataset_AgeDB30)))  
    
    if dataset == 'CASIA':
        root_casia = os.path.join(DATASETS_DIR, 'CASIA')
        root_legacy = 'data_set/CASIA'
        root_legacy2 = 'data_set/CASIA_Webface_Image'
        if os.path.isdir(root_casia):
            root = root_casia
            file_list = os.path.join(root_casia, 'webface_align_112.txt')
        elif os.path.isdir(root_legacy):
            root = root_legacy
            file_list = os.path.join(root_legacy, 'webface_align_112.txt')
        else:
            root = root_legacy2
            file_list = os.path.join(root_legacy2, 'webface_align_112.txt')
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':
        root_ms1m = os.path.join(DATASETS_DIR, 'ms1m-arcface')
        root_emore = os.path.join(DATASETS_DIR, 'faces_emore_images')
        root_m1m = os.path.join(DATASETS_DIR, 'm1m-arcface-centered')
        root_legacy = 'data_set/faces_emore_images'
        if os.path.isdir(root_ms1m):
            root = root_ms1m
            file_list = os.path.join(root_ms1m, 'ms1m_arcface_align_112.txt')
        elif os.path.isdir(root_emore):
            root = root_emore
            file_list = os.path.join(root_emore, 'faces_emore_align_112.txt')
        elif os.path.isdir(root_m1m):
            root = root_m1m
            file_list = os.path.join(root_m1m, 'faces_emore_align_112.txt')
        else:
            root = root_legacy
            file_list = os.path.join(root_legacy, 'faces_emore_align_112.txt')
        dataset_train = MS1M(root, file_list, transform=transform) 
    
    else:
        raise NameError('no training data exist!')
    
    print('Training data: root={!r}, file_list={!r}'.format(os.path.abspath(root), os.path.abspath(file_list)))
    if 'ms1m_arcface_align_112.txt' in file_list and not os.path.isfile(file_list):
        print('  -> Annotation missing? Generate with: python data_set/anno_generation.py --dataset ms1m-arcface')
    
    eval_kw = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                   persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor if num_workers else None)
    dataloaders = {
        'LFW': data.DataLoader(dataset_LFW, **eval_kw),
        'CFP_FP': data.DataLoader(dataset_CFP_FP, **eval_kw),
        'AgeDB30': data.DataLoader(dataset_AgeDB30, **eval_kw),
    }
    dataset = {'LFW': dataset_LFW, 'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    dataset_sizes = {'LFW': len(dataset_LFW), 'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    if use_dali and torch.cuda.is_available():
        from data_set.dali_loader import create_dali_train_loader
        train_iter, train_class_nums, train_size = create_dali_train_loader(
            file_root=root,
            file_list=file_list,
            batch_size=batch_size,
            device_id=device_id,
            num_threads=4,
            crop_size=112,
        )
        dataloaders['train'] = train_iter
        dataset_sizes['train'] = train_size
        dataset['train'] = type('Dummy', (), {'class_nums': train_class_nums})()
        dali_loader_args = (root, file_list, batch_size, device_id, 4, 112)
        print('training data loaded (DALI GPU decode), validation data loaded')
    else:
        persistent = num_workers > 0
        train_kw = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=persistent, prefetch_factor=prefetch_factor if num_workers else None)
        dataloaders['train'] = data.DataLoader(dataset_train, **train_kw)
        dataset_sizes['train'] = len(dataset_train)
        dataset['train'] = dataset_train
        dali_loader_args = None
        print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset, dali_loader_args

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Face_Detection_Training')
    parser.add_argument('--dataset', type=str, default='Faces_emore', help='Training dataset: CASIA, Faces_emore')
    parser.add_argument('--feature_dim', type=int, default=512, help='the feature dimension output')
    parser.add_argument('--batch_size', type=int, default=192, help='batch size (use 128-192 with DALI on 12GB GPU)')
    parser.add_argument('--epoch', type=int, default=12, help='number of epoches for training')
    parser.add_argument('--method', type=str, default='l2_distance', 
                            help='methold to evaluate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    parser.add_argument('--num_workers', type=int, default=12, help='DataLoader num_workers')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch_factor (batches per worker)')
    parser.add_argument('--use_dali', action='store_true', help='Use NVIDIA DALI for GPU decode (faster training)')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    device_id = 0 if device.type == 'cuda' else -1
    dataloaders, dataset_sizes, dataset, dali_loader_args = load_data(
        args.batch_size, dataset=args.dataset, num_workers=args.num_workers, pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor, use_dali=args.use_dali, device_id=device_id)
    model = MobileFaceNet(args.feature_dim).to(device)  # embeding size is 512 (feature vector)
    print('MobileFaceNet face detection model loaded')
    margin = Arcface(embedding_size=args.feature_dim, classnum=int(dataset['train'].class_nums),  s=64., m=0.5).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.01, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 8, 10], gamma=0.3) 
    start = time.time()
    ## save logging and weights
    train_logging_file = 'train_{}_logging.txt'.format(args.dataset)
    test_logging_file = 'test_{}_logging.txt'.format(args.dataset)
    save_dir = 'saving_{}_ckpt'.format(args.dataset)
    already_existed = os.path.exists(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if already_existed:
        print('Checkpoint dir already exists; writing into:', save_dir)

    best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB30': 0.0}
    best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB30': 0}
    total_iters = 0
    train_size = dataset_sizes['train']
    batches_per_epoch = math.ceil(train_size / args.batch_size) if args.use_dali else len(dataloaders['train'])
    total_iters_planned = args.epoch * batches_per_epoch
    print('Training config: {} epochs, {} batches/epoch, {} total iters, batch_size={}, num_workers={}'
          .format(args.epoch, batches_per_epoch, total_iters_planned, args.batch_size, args.num_workers))
    print('Logs: {} | {} | checkpoints: {}'.format(train_logging_file, test_logging_file, save_dir))
    print('-' * 10)
    for epoch in range(args.epoch):
        model.train()
        if dali_loader_args is not None:
            from data_set.dali_loader import create_dali_train_loader
            train_iter, _, _ = create_dali_train_loader(*dali_loader_args)
            train_batches = train_iter
        else:
            train_batches = dataloaders['train']
        since = time.time()
        since_100 = time.time()
        for batch_idx, det in enumerate(train_batches):
            img, label = det[0].to(device), det[1].to(device)
            optimizer_ft.zero_grad()
            with torch.set_grad_enabled(True):
                raw_logits = model(img)
                output = margin(raw_logits, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer_ft.step()
            total_iters += 1
            if total_iters % 100 == 0:
                _, preds = torch.max(output.data, 1)
                total = label.size(0)
                correct = (preds.cpu().numpy() == label.data.cpu().numpy()).sum()
                time_per_iter = (time.time() - since_100) / 100
                since_100 = time.time()
                lr = optimizer_ft.param_groups[0]['lr']
                remaining = total_iters_planned - total_iters
                eta_sec = remaining * time_per_iter
                eta_str = '{:.0f}m {:.0f}s'.format(eta_sec // 60, eta_sec % 60) if remaining > 0 else '0s'
                print('Epoch {}/{}, batch {}/{}, iters {:>6d}/{}, loss: {:.4f}, train_acc: {:.4f}, {:.3f} s/iter, lr: {}, ETA: {}'
                      .format(epoch, args.epoch - 1, batch_idx + 1, batches_per_epoch, total_iters, total_iters_planned,
                              loss.item(), correct / total, time_per_iter, lr, eta_str))
                with open(train_logging_file, 'a') as f:
                    f.write('Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, lr: {}, ETA: {}\n'
                            .format(epoch, args.epoch - 1, total_iters, loss.item(), correct / total, time_per_iter, lr, eta_str))
            if total_iters % 3000 == 0:
                ckpt_model = os.path.join(save_dir, 'Iter_%06d_model.ckpt' % total_iters)
                ckpt_margin = os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters)
                torch.save({'iters': total_iters, 'net_state_dict': model.state_dict()}, ckpt_model)
                torch.save({'iters': total_iters, 'net_state_dict': margin.state_dict()}, ckpt_margin)
                print('  [Checkpoint] iter {} -> {} & {}'.format(total_iters, ckpt_model, ckpt_margin))
                model.eval()
                for phase in ['LFW', 'CFP_FP', 'AgeDB30']:
                    featureLs, featureRs = getFeature(model, dataloaders[phase], device, flip=args.flip)
                    ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method=args.method)
                    acc_pct = np.mean(ACCs) * 100
                    print('  [Eval] {} acc: {:.2f}% (avg threshold: {:.4f})'.format(phase, acc_pct, np.mean(threshold)))
                    if best_acc[phase] <= acc_pct:
                        best_acc[phase] = acc_pct
                        best_iters[phase] = total_iters
                    with open(test_logging_file, 'a') as f:
                        f.write('Epoch {}/{}, {} average acc: {:.4f} average threshold: {:.4f}\n'
                                .format(epoch, args.epoch - 1, phase, acc_pct, np.mean(threshold)))
                model.train()
        exp_lr_scheduler.step()
                        
    time_elapsed = time.time() - start  
    print('Finally Best Accuracy: LFW: {:.4f} in iters: {}, CFP_FP: {:.4f} in iters: {} and AgeDB-30: {:.4f} in iters: {}'.format(
        best_acc['LFW'], best_iters['LFW'], best_acc['CFP_FP'], best_iters['CFP_FP'], best_acc['AgeDB30'], best_iters['AgeDB30']))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        
