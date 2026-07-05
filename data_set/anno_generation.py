#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the annotation file for MS1M/faces_emore-style training (path label per line).
Expects root to contain identity subdirs (id_0, id_1, ...) with images inside.

Usage:
  python data_set/anno_generation.py --dataset ms1m-arcface   # ~/Datasets/ms1m-arcface -> ms1m_arcface_align_112.txt
  python data_set/anno_generation.py --root /path/to/images --file /path/to/list.txt
"""
import os
import argparse

DATASETS_DIR = os.path.expanduser('~/Datasets')

parser = argparse.ArgumentParser(description='Annotation file generator for MS1M/faces_emore-style data')
parser.add_argument('--dataset', type=str, choices=['ms1m-arcface', 'faces_emore'], default=None,
                   help='Preset: ms1m-arcface -> ~/Datasets/ms1m-arcface + ms1m_arcface_align_112.txt; faces_emore -> faces_emore_images + faces_emore_align_112.txt')
parser.add_argument('--root', type=str, default=None, help='Image root (identity subdirs). Overridden by --dataset if set.')
parser.add_argument('--file', type=str, default=None, help='Output annotation path. Overridden by --dataset if set.')
args = parser.parse_args()

if args.dataset == 'ms1m-arcface':
    imgdir = os.path.join(DATASETS_DIR, 'ms1m-arcface')
    list_txt_file = os.path.join(imgdir, 'ms1m_arcface_align_112.txt')
elif args.dataset == 'faces_emore':
    imgdir = 'faces_emore_images'
    list_txt_file = os.path.join(imgdir, 'faces_emore_align_112.txt')
else:
    imgdir = args.root or 'faces_emore_images'
    list_txt_file = args.file or os.path.join(imgdir, 'faces_emore_align_112.txt')
docs = [f for f in os.listdir(imgdir) if not f.startswith('.') and os.path.isdir(os.path.join(imgdir, f))]
docs.sort()

label = 0
with open(list_txt_file, 'w') as f:
    for name in docs:
        print('writing name:', name)
        image_folder = imgdir+'/'+name
        files = [x for x in os.listdir(image_folder) if not x.startswith('.')]
        files.sort()

        for fn in files:
            txt_name = os.path.join(name, fn)
            f.write(txt_name+' '+str(label)+'\n')

        label += 1

print('writing finished')


