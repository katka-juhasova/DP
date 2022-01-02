import argparse
import os
import tensorflow as tf
import pointnet.utils as utils
import pointnet.pointnet as pointnet
from pointnet.generator import Generator
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, help='Trained weights path')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training [default: 32]')
args = parser.parse_args()


WEIGHTS = args.weights
NUM_POINT = args.num_point
NUM_CLASS = 40
BATCH_SIZE = args.batch_size


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

test_files = utils.get_data_files(TEST_FILES)
shapes_file = os.path.join(DATA_DIR, 'shape_names.txt')

# Read shape names
with open(shapes_file) as f:
    shapes = f.read().splitlines()

# Test data generator
test_generator = Generator(test_files, num_point=NUM_POINT,
                           batch_size=BATCH_SIZE, jitter=False,
                           rotate=False, shuffle=False)

model_name = WEIGHTS.split('/')[0].split('_')[-1]

# Load weights nad model
# 2021-10-27_07:38:35_PointNet-1zy4zmyd/
# model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5
weights_path = os.path.join(BASE_DIR, 'models', WEIGHTS)

model = pointnet.get_model(num_point=NUM_POINT, num_class=NUM_CLASS)
model.load_weights(weights_path)

model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

# Quantitative evaluation
# NOTE: Original PointNet accuracy: 89.2%
val_loss, val_acc = model.evaluate(test_generator, verbose=0)

print(model_name)
print('loss: {:.2f}, acc: {:.2f}%'.format(val_loss, 100 * val_acc))
