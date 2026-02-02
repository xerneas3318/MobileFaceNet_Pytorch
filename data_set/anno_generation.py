#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:36:12 2019
Generate the annotation file for webface_align_112

@author: AIRocker
"""
import os
import argparse

parser = argparse.ArgumentParser(description='Annotation file generator')
parser.add_argument('--root', type=str, default='faces_emore_images', help='image data folder path')
parser.add_argument('--file', type=str, default='faces_emore_images/faces_emore_align_112.txt', help='anno file path')
args = parser.parse_args()

imgdir = args.root
list_txt_file = args.file
docs = [f for f in os.listdir(imgdir) if not f.startswith('.')]
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


