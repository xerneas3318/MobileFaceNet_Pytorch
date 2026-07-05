#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:57:15 2019
Cam demo

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import argparse
import re
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.util import *
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import cv2
import time

def resize_image(img, scale):
    """
        resize image
    """
    if img is None or scale is None:
        return img
    height, width, channel = img.shape
    if height == 0 or width == 0:
        return img
    if scale <= 0:
        scale = 1.0
    new_height = max(1, int(height * scale))     # resized new height
    new_width = max(1, int(width * scale))       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

def get_latest_model_ckpt(ckpt_dir):
    if not ckpt_dir:
        return None
    ckpt_dir = os.path.abspath(os.path.expanduser(ckpt_dir))
    if not os.path.isdir(ckpt_dir):
        return None
    pattern = re.compile(r"Iter_(\d+)_model\.ckpt$")
    latest_iter = -1
    latest_path = None
    for name in os.listdir(ckpt_dir):
        match = pattern.match(name)
        if not match:
            continue
        iters = int(match.group(1))
        if iters > latest_iter:
            latest_iter = iters
            latest_path = os.path.join(ckpt_dir, name)
    return latest_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face detection demo')
    parser.add_argument('-th','--threshold',help='threshold score to decide identical faces',default=80, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true", default= False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true", default= False)
    parser.add_argument("-c", "--score", help="show match similarity 0-100 (higher = better match to facebank identity)", action="store_true", default=True)
    parser.add_argument("--detect_only", action="store_true",
                        help="only detect faces; show 'Person' per face, no facebank loading or recognition")
    parser.add_argument("--facebank_path", dest="facebank_path", default="facebank",
                        help="Directory containing one subfolder per identity, or flat image folder. Embeddings saved as facebank.pth + names.npy.")
    parser.add_argument("--facebank_name", dest="facebank_name", default=None,
                        help="When facebank_path is a flat folder of one person, set this to their name (e.g. Rian). All images are treated as that identity.")
    parser.add_argument("--scale", dest='scale', help="input frame scale to accurate the speed", default=0.5, type=float)
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default=20, type=int)
    parser.add_argument('--weights', dest='weights', default=None,
                        help='path to MobileFaceNet weights (default: latest in ckpt_dir or Weights/MobileFace_Net)')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='saving_Faces_emore_ckpt',
                        help='directory to search for latest Iter_*_model.ckpt')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights_path = args.weights
    if weights_path is None:
        weights_path = get_latest_model_ckpt(args.ckpt_dir) or 'Weights/MobileFace_Net'
    weights_path = os.path.expanduser(weights_path)
    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    state = torch.load(weights_path, map_location=lambda storage, loc: storage)
    if isinstance(state, dict):
        if "net_state_dict" in state:
            state = state["net_state_dict"]
        elif "model_state" in state:
            state = state["model_state"]
    detect_model.load_state_dict(state)
    print('MobileFaceNet face detection model generated')
    print('Using weights:', weights_path)
    detect_model.eval()

    targets, names = None, None
    if not args.detect_only:
        if args.update:
            targets, names = prepare_facebank(detect_model, path=args.facebank_path, tta=args.tta, single_identity_name=args.facebank_name)
            targets = targets.to(device)
            print('facebank updated')
        else:
            targets, names = load_facebank(path=args.facebank_path, device=device)
            print('facebank loaded')
    else:
        print('detect_only: showing "Person" for each face (no recognition)')

    cap = cv2.VideoCapture(0)
    while True:
        isSuccess, frame = cap.read()
        if isSuccess:
            if frame is None or frame.size == 0:
                continue
            try:
                start_time = time.time()
                input = resize_image(frame, args.scale)
                bboxes, landmarks = create_mtcnn_net(input, args.mini_face, device, p_model_path='MTCNN/weights/pnet_Weights',
                                                     r_model_path='MTCNN/weights/rnet_Weights',
                                                     o_model_path='MTCNN/weights/onet_Weights')

                bboxes = np.asarray(bboxes)
                landmarks = np.asarray(landmarks)
                n_bboxes = len(bboxes) if bboxes.ndim >= 1 else 0
                n_landmarks = landmarks.shape[0] if landmarks.ndim >= 1 and landmarks.size > 0 else 0

                if n_bboxes > 0:
                    if bboxes.ndim == 1:
                        bboxes = bboxes.reshape(1, -1)
                    bboxes = bboxes / args.scale
                if n_landmarks > 0 and n_landmarks == n_bboxes:
                    landmarks = landmarks / args.scale
                else:
                    landmarks = np.array([])

                faces = Face_alignment(frame, default_square=True, landmarks=landmarks)

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype('utils/simkai.ttf', 30)

                if len(faces) > 0 and len(faces) == len(bboxes):
                    if args.detect_only:
                        for i, b in enumerate(bboxes):
                            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='blue', width=5)
                            draw.text((int(b[0]), int(b[1] - 25)), 'Person', fill=(255, 255, 0), font=font)
                    else:
                        embs = []
                        test_transform = trans.Compose([
                            trans.ToTensor(),
                            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                        for img in faces:
                            if args.tta:
                                mirror = cv2.flip(img, 1)
                                emb = detect_model(test_transform(img).to(device).unsqueeze(0))
                                emb_mirror = detect_model(test_transform(mirror).to(device).unsqueeze(0))
                                embs.append(l2_norm(emb + emb_mirror))
                            else:
                                embs.append(detect_model(test_transform(img).to(device).unsqueeze(0)))

                        source_embs = torch.cat(embs)
                        diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0)
                        dist = torch.sum(torch.pow(diff, 2), dim=1)
                        minimum, min_idx = torch.min(dist, dim=1)
                        min_idx[minimum > ((args.threshold - 156) / (-80))] = -1
                        results = min_idx
                        score_100 = torch.clamp(minimum * -80 + 156, 0, 100)

                        for i, b in enumerate(bboxes):
                            is_match = int(results[i]) >= 0
                            box_color = (0, 255, 0) if is_match else (255, 0, 0)  # green = right person, red = wrong
                            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=box_color, width=5)
                            if args.score:
                                draw.text((int(b[0]), int(b[1] - 25)),
                                          names[results[i] + 1] + ' score:{:.0f}'.format(score_100[i].item()),
                                          fill=(255, 255, 0), font=font)
                            else:
                                draw.text((int(b[0]), int(b[1] - 25)), names[results[i] + 1],
                                          fill=(255, 255, 0), font=font)

                    for p in landmarks:
                        for i in range(5):
                            draw.ellipse([(p[i] - 2.0, p[i + 5] - 2.0), (p[i] + 2.0, p[i + 5] + 2.0)], outline='blue')
                else:
                    for b in bboxes:
                        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='blue', width=5)
                        if args.detect_only:
                            draw.text((int(b[0]), int(b[1] - 25)), 'Person', fill=(255, 255, 0), font=font)

                FPS = 1.0 / (time.time() - start_time)
                draw.text((10, 10), 'FPS: {:.1f}'.format(FPS), fill=(0, 0, 0), font=font)
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            except Exception as e:
                print('detect error:', e)

            cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
