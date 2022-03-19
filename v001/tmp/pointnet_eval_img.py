import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import v001.pointnet.pointnet_utils as utils
import v001.pointnet.pointnet_model as pointnet
from v001.pointnet.pointnet_generator import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True,
                    help='Trained weights path')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024/2048] [default: 1024]')
args = parser.parse_args()


WEIGHTS = args.weights
NUM_POINT = args.num_point
NUM_CLASS = 40
BATCH_SIZE = 2048

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'ModelNet40')
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

# Load weights and model
# 2021-10-27_07:38:35_PointNet-1zy4zmyd/
# model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5
model = pointnet.get_model(num_point=NUM_POINT, num_class=NUM_CLASS)
model.load_weights(WEIGHTS)

# Qualitative evaluation
data, labels = test_generator.__getitem__(0)
preds = model.predict(data)
preds = tf.math.argmax(preds, -1)

items = ["monitor", "chair", "plant", "piano", "desk",
         "toilet", "bookshelf", "lamp", "airplane", "bathtub"]
num_items = len(items)
items_idx = list()

# Collect indices of all required items
for i, label in enumerate(labels):
    item = shapes[label]
    if item in items:
        items_idx.append(i)
        items.remove(item)

        if not items_idx:
            break

# Plot points with predicted class and label
plt.style.use('seaborn-deep')
fig = plt.figure(figsize=(20, 8))
rows = 2
cols = num_items // rows

camera = [[0, -90], [25, -50], [20, -60], [0, -70], [30, -35],
          [0, 180], [10, 180], [0, 180], [0, -90], [0, 0]]

for i, item_idx in enumerate(items_idx):
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    ax.scatter(data[item_idx, :NUM_POINT, 0],
               data[item_idx, :NUM_POINT, 2],
               data[item_idx, :NUM_POINT, 1],
               alpha=0.35)
    ax.set_title(
        "pred: {:}, label: {:}".format(
            shapes[preds[item_idx].numpy()], shapes[labels[item_idx]]
        ),
        fontsize=16
    )
    ax.set_axis_off()
    ax.set_xlim3d(-0.7, 0.7)
    ax.set_ylim3d(-0.7, 0.7)
    ax.set_zlim3d(-0.7, 0.7)
    ax.view_init(camera[i][0], camera[i][1])

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
# plt.show()

# Save the figure
model_name = os.path.basename(os.path.dirname(WEIGHTS))
model_name = '_'.join(model_name.split('_')[2:])
fig_path = os.path.join(BASE_DIR, '..', 'results', model_name + '.png')
plt.savefig(fig_path)
