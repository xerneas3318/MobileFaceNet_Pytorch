#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Generate the face bank

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
from MTCNN import create_mtcnn_net
from utils.align_trans import *
import numpy as np
from torchvision import transforms as trans
import torch
from face_model import MobileFaceNet, l2_norm
from pathlib import Path
import cv2

test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def _process_image_to_emb(model, img, tta):
    """Run model on one image (numpy HWC). Returns tensor (1, 512) or None if failed."""
    if img is None or not hasattr(img, 'shape') or len(img.shape) != 3:
        return None
    if img.shape != (112, 112, 3):
        bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                             p_model_path='MTCNN/weights/pnet_Weights',
                                             r_model_path='MTCNN/weights/rnet_Weights',
                                             o_model_path='MTCNN/weights/onet_Weights')
        img = Face_alignment(img, default_square=True, landmarks=landmarks)
        if not img or len(img) == 0:
            return None
        img = img[0]
    with torch.no_grad():
        if tta:
            mirror = cv2.flip(img, 1)
            emb = model(test_transform(img).to(device).unsqueeze(0))
            emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
            return l2_norm(emb + emb_mirror)
        return model(test_transform(img).to(device).unsqueeze(0))

def prepare_facebank(model, path='facebank', tta=True, single_identity_name=None):
    """
    Build facebank from a folder. single_identity_name: if set and folder is flat (no subdirs),
    treat all images as one person with this name (one averaged embedding).
    """
    model.eval()
    embeddings = []
    names = ['']
    data_path = Path(path)
    if not data_path.is_dir():
        raise FileNotFoundError('Facebank path is not a directory: {!r}'.format(path))

    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        # One subfolder per identity (original layout)
        for doc in sorted(subdirs):
            embs = []
            for fname in listdir_nohidden(doc):
                if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image_path = os.path.join(doc, fname)
                img = cv2.imread(image_path)
                emb = _process_image_to_emb(model, img, tta)
                if emb is not None:
                    embs.append(emb)
            if len(embs) == 0:
                continue
            embedding = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(embedding)
            names.append(doc.name)
    else:
        # Flat layout: collect (emb, stem) for each valid image
        embs = []
        stems = []
        for fname in sorted(listdir_nohidden(data_path)):
            if fname in ('facebank.pth', 'names.npy') or Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_path = os.path.join(data_path, fname)
            if not os.path.isfile(image_path):
                continue
            img = cv2.imread(image_path)
            emb = _process_image_to_emb(model, img, tta)
            if emb is not None:
                embs.append(emb)
                stems.append(Path(fname).stem)
        if embs:
            if single_identity_name is not None:
                # All images = one person (e.g. "Rian")
                embedding = torch.cat(embs).mean(0, keepdim=True)
                embeddings.append(embedding)
                names.append(single_identity_name)
            else:
                # One image = one identity (filename stem as name)
                embeddings.extend(embs)
                names = [''] + stems

    if len(embeddings) == 0:
        raise ValueError(
            'No valid identities found in {!r}. '
            'Expected either: (1) one subfolder per person with images inside, or '
            '(2) image files directly in the folder (one identity per image).'.format(path)
        )
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(path, 'facebank.pth'))
    np.save(os.path.join(path, 'names'), names)
    return embeddings, names

def load_facebank(path='facebank', device=None):
    """Load facebank embeddings and names. If device is set (e.g. cuda:0), embeddings are moved to device for GPU comparison."""
    data_path = Path(path)
    pth = data_path / 'facebank.pth'
    npy = data_path / 'names.npy'
    if not pth.is_file():
        raise FileNotFoundError(
            'Facebank not found at {!r}. Run the demo with --update to build it from images in that folder. '
            'Folder should contain either: one subfolder per person, or image files (one identity per image).'.format(str(pth))
        )
    embeddings = torch.load(pth, map_location='cpu')
    if device is not None:
        embeddings = embeddings.to(device)
    names = np.load(npy, allow_pickle=True)
    return embeddings, names

if __name__ == '__main__':

    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(
        torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    embeddings, names = prepare_facebank(detect_model, path = 'facebank', tta = True)
    print(embeddings.shape)
    print(names)






