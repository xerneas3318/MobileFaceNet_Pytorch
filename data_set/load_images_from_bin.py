#!/usr/bin/env python
# encoding: utf-8
'''
@desc: For AgeDB-30 and CFP-FP test dataset, we use the mxnet binary file provided by insightface, this is the tool to restore
       the aligned images from mxnet binary file.
       Bin mode: no mxnet needed (uses OpenCV). Rec mode: requires mxnet (use Python 3.9–3.11 + mxnet==1.2.1).
'''

import cv2
import numpy as np
import os
import pickle
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

'''
For train dataset, insightface provide a mxnet .rec file; use Python 3.9–3.11 and mxnet for rec mode.
'''


def load_mx_rec(rec_path, save_path):
    import mxnet as mx
    save_path = os.path.join(rec_path, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = os.path.join(save_path, str(label).zfill(6))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        cv2.imwrite(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), img)


def load_image_from_bin(bin_path, save_dir, pair_filename='pair.txt', pair_dir=None):
    """Convert .bin (pickle of encoded images + issame_list) to images. No mxnet required."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_pair_dir = pair_dir or os.path.dirname(save_dir)
    os.makedirs(out_pair_dir, exist_ok=True)
    pair_path = os.path.join(out_pair_dir, pair_filename)
    with open(bin_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0)
            data = pickle.load(f, encoding='bytes')
    bins, issame_list = data[0], data[1]
    with open(pair_path, 'w') as pair_file:
        for idx in tqdm(range(len(bins))):
            _bin = bins[idx]
            if isinstance(_bin, (bytes, bytearray)):
                buf = np.frombuffer(_bin, dtype=np.uint8)
            else:
                buf = np.array(_bin, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f'Failed to decode image at index {idx} in {bin_path}')
            cv2.imwrite(os.path.join(save_dir, str(idx + 1).zfill(5) + '.jpg'), img)
            if idx % 2 == 0:
                label = 1 if issame_list[idx // 2] else -1
                pair_file.write(
                    str(idx + 1).zfill(5) + '.jpg' + ' ' + str(idx + 2).zfill(5) + '.jpg' + ' ' + str(label) + '\n'
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mxnet/bin file to image')
    parser.add_argument('--rec', type=str, default='download_rec/faces_emore', help='rec dir path (mode=rec)')
    parser.add_argument('--save_rec', type=str, default='faces_emore_images', help='save dir within rec path')
    parser.add_argument('--bin', type=str, default='data_set/lfw.bin', help='.bin file path (mode=bin)')
    parser.add_argument('--save_bin', type=str, default='data_set/LFW/lfw_align_112', help='output image dir')
    parser.add_argument('--pair_file', type=str, default='pairs.txt', help='pair list filename (e.g. pairs.txt, cfp_fp_pair.txt)')
    parser.add_argument('--mode', type=str, default='bin', choices=('bin', 'rec'), help='convert bin or rec')
    args = parser.parse_args()

    if args.mode == 'rec':
        load_mx_rec(args.rec, args.save_rec)
    else:
        load_image_from_bin(args.bin, args.save_bin, pair_filename=args.pair_file)
