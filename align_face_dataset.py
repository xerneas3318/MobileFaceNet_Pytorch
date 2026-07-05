#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align and crop faces in a dataset using MTCNN + similarity transform.
Reads images from an input folder (face not necessarily centered), detects the face,
aligns to 5 landmarks, crops 112x112, and saves to an output folder.
Use this before building a facebank or training when faces are not centered.

Usage:
  python align_face_dataset.py --input Labeled_Rians_in_the_wild --output Labeled_Rians_in_the_wild_aligned
  python align_face_dataset.py -i path/to/images -o path/to/aligned [--mini_face 20]
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MTCNN'))

import numpy as np
import cv2
import torch
from MTCNN import create_mtcnn_net
from utils.align_trans import Face_alignment

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MTCNN_WEIGHTS = {
    'p_model_path': 'MTCNN/weights/pnet_Weights',
    'r_model_path': 'MTCNN/weights/rnet_Weights',
    'o_model_path': 'MTCNN/weights/onet_Weights',
}


def main():
    parser = argparse.ArgumentParser(description='Align and crop faces in a dataset with MTCNN')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder containing images (faces need not be centered)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for 112x112 aligned face crops (same filenames)')
    parser.add_argument('--mini_face', type=int, default=20,
                        help='MTCNN minimum face size (default 20); decrease for small faces')
    parser.add_argument('--skip_no_face', action='store_true', default=True,
                        help='Skip images where no face is detected (default True)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print('Error: input is not a directory:', input_dir)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        and f.name not in ('facebank.pth', 'names.npy')
    ]
    image_files.sort(key=lambda p: p.name)
    total = len(image_files)
    if total == 0:
        print('No image files found in', input_dir)
        sys.exit(0)

    print('MTCNN + align: {} images from {} -> {}'.format(total, input_dir, output_dir))
    ok, skipped, failed = 0, 0, 0

    for idx, path in enumerate(image_files):
        img = cv2.imread(str(path))
        if img is None:
            print('  [skip] cannot read:', path.name)
            failed += 1
            continue
        try:
            bboxes, landmarks = create_mtcnn_net(
                img, args.mini_face, device, **MTCNN_WEIGHTS
            )
            bboxes = np.asarray(bboxes) if bboxes is not None else np.array([])
            landmarks = np.asarray(landmarks) if landmarks is not None else np.array([])
            n = bboxes.size if bboxes.ndim >= 1 else 0
            if n == 0:
                if args.skip_no_face:
                    skipped += 1
                    if (idx + 1) % 20 == 0 or idx == 0:
                        print('  [skip no face]', path.name)
                else:
                    # Save original crop or a black 112x112 placeholder
                    out_path = output_dir / path.name
                    small = cv2.resize(img, (112, 112))
                    cv2.imwrite(str(out_path), small)
                    ok += 1
                continue
            faces = Face_alignment(img, default_square=True, landmarks=landmarks)
            if not faces:
                skipped += 1
                continue
            # Use first (or largest) face
            face = faces[0]
            if len(faces) > 1:
                # Prefer largest by area
                h, w = img.shape[:2]
                best = 0
                best_area = 0
                for i, f in enumerate(faces):
                    a = f.shape[0] * f.shape[1]
                    if a > best_area:
                        best_area = a
                        best = i
                face = faces[best]
            out_path = output_dir / path.name
            cv2.imwrite(str(out_path), face)
            ok += 1
        except Exception as e:
            print('  [error]', path.name, e)
            failed += 1
        if (idx + 1) % 50 == 0:
            print('  {}/{} done (ok={}, skip={}, fail={})'.format(idx + 1, total, ok, skipped, failed))

    print('Done. ok={}, skipped={}, failed={}'.format(ok, skipped, failed))
    print('Aligned faces saved to:', output_dir)


if __name__ == '__main__':
    main()
