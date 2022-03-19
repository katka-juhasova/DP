import argparse
import os
import pandas as pd
import cv2 as cv
import numpy as np
import h5py
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, required=True, 
                    help='Src dir with raw symbols data to preprocess')
parser.add_argument('--dest_dir', type=str, required=True,
                    help='Dest dir for preprocessed symbols data')
parser.add_argument('--num_point', type=int, default=2048,
                    help='Number of points to sample from the raw image \
                    [default: 2048]')
parser.add_argument('--limit', type=int, default=2000,
                    help='Limit for minimal size of the raw image \
                    [default: 2000]')
args = parser.parse_args()


# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, args.src_dir)
DEST_DIR = os.path.join(BASE_DIR, args.dest_dir)
NUM_POINT = args.num_point
LIMIT = args.limit


data_dirs = list()
symbols = list()

for root, dirs, files in os.walk(SRC_DIR):
    depth = root[len(SRC_DIR) + len(os.path.sep):].count(os.path.sep)
    if depth == 1:
        data_dirs.append(root)

for data_dir in tqdm(data_dirs):
    height_map_file = os.path.join(data_dir, 'heightMap.png')
    glyphs_file = os.path.join(data_dir, 'Glyphs.csv')

    glyphs = pd.read_csv(glyphs_file)
    glyphs = glyphs[glyphs.Width * glyphs.Height >= LIMIT]

    height_map = cv.imread(height_map_file, 0)

    for index, row in glyphs.iterrows():
        img = cv.imread(os.path.join(data_dir, row['Mask']), 0)

        ylim1 = row['Y']
        ylim2 = row['Y'] + row['Height']
        ydiff = 0
        xlim1 = row['X']
        xlim2 = row['X'] + row['Width']
        xdiff = 0
        
        h_map = height_map[ylim1:ylim2, xlim1:xlim2]

        if img.shape[0] != h_map.shape[0]:
            ydiff = img.shape[0] + row['Y']
        if img.shape[1] != h_map.shape[1]:
            xdiff = img.shape[1] - h_map.shape[1]
            
        if ydiff and xdiff:
            tmp1 = height_map[ylim1:, xlim1:xlim2]
            tmp2 = height_map[ylim1:, 0:xdiff]
            concat1 = np.concatenate((tmp1, tmp2), axis=1)
            tmp3 = height_map[0:ydiff, xlim1:xlim2]
            tmp4 = height_map[0:ydiff, 0:xdiff]
            concat2 = np.concatenate((tmp3, tmp4), axis=1)
            h_map = np.concatenate((concat1, concat2), axis=0)
        elif ydiff:
            h_map = height_map[ylim1:, xlim1:xlim2]
            tmp = height_map[0:ydiff, xlim1:xlim2]
            h_map = np.concatenate((h_map, tmp), axis=0)
        elif xdiff:
            tmp = height_map[ylim1:ylim2, 0:xdiff]
            h_map = np.concatenate((h_map, tmp), axis=1)

        # Masking & sampling
        mask_coords = np.argwhere(img == 255)
        idx = np.random.randint(mask_coords.shape[0], size=NUM_POINT)
        selection = mask_coords[idx, :]

        # Object coordinates
        x = np.expand_dims(np.array(selection[:, 1]), axis=-1)
        y = np.expand_dims(np.array(selection[:, 0]), axis=-1)
        z = h_map[y, x]
        y = -y

        # Normalize object into unit sphere
        obj = np.concatenate((x, y, z), axis=1)

        x_size = np.max(x) - np.min(x)
        y_size = np.max(y) - np.min(y)
        z_size = np.max(z) - np.min(z)

        x_center = x_size / 2 + np.min(x)
        y_center = y_size / 2 + np.min(y)
        z_center = z_size / 2 + np.min(z)
        center = np.array([x_center, y_center, z_center])
        new_obj = (obj - center) / (np.max([x_size, y_size, z_size]) / 2)

        symbols.append(new_obj)
        
        # plt.style.use('seaborn-deep')
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(new_obj[:, 0], new_obj[:, 1], new_obj[:, 2], alpha=0.35)
        # ax.set_xlim3d(-1, 1)
        # ax.set_ylim3d(-1, 1)
        # ax.set_zlim3d(-1, 1)

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

symbols = np.array(symbols, dtype=np.float32)
dest_file = os.path.join(DEST_DIR, 'symbols.h5')
with h5py.File(dest_file, 'w') as f:
    h5_symbols = f.create_dataset('symbols', data=symbols)

print('Data successfully saved to {}'.format(dest_file))
