import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import corsnet.model as corsnet
from corsnet.generator import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='Data dir containing train & test dataset')
parser.add_argument('--weights', type=str, required=True,
                    help='Trained weights path')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for evaluation [default: 128]')
args = parser.parse_args()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, args.data_dir)
WEIGHTS = os.path.join(BASE_DIR, args.weights)
NUM_POINT = args.num_point
BATCH_SIZE = args.batch_size


model_name = os.path.basename(os.path.dirname(WEIGHTS))
model_name = '_'.join(model_name.split('_')[2:])
test_file = os.path.join(DATA_DIR, 'test_dataset.h5')
# Test data generator
test_generator = Generator(test_file,
                           batch_size=BATCH_SIZE,
                           jitter=False,
                           shuffle=False,
                           shuffle_points=False)

# Load model and weights
model = corsnet.get_model(num_point=NUM_POINT)
model.load_weights(WEIGHTS)

# Qualitative evaluation
[src, temp], t_matrix = test_generator.__getitem__(0)
preds = model.predict([src, temp])

items_idx = [10, 11, 23, 29, 34, 45, 50, 51, 54, 43, 57, 59, 14, 72, 115, 97]

# Plot results
plt.style.use('seaborn-deep')
fig = plt.figure(figsize=(16, 14))
rows = 4
cols = 4

camera = [[10, -50], [10, -90], [10, -90], [-90, 0],
          [0, 0], [0, 0], [0, -90], [0, 0],
          [0, 0], [0, 0], [90, 0], [0, 0],
          [0, 45], [0, 35], [0, 0], [20, -145]]

limits = [[-0.6, 0.8], [-0.4, 0.8], [-0.35, 0.9], [-0.35, 1.1],
          [-0.6, 0.9], [-0.2, 1.0], [-0.3, 1.1], [-0.6, 0.75],
          [-0.35, 0.95], [-0.4, 0.9], [-1.5, 0.8], [-0.3, 0.9],
          [-0.3, 1.0], [-0.1, 1.0], [-0.45, 1.1], [-0.6, 0.7]]

for i, item_idx in enumerate(items_idx):
    x_src = src[item_idx, :, 0]
    y_src = src[item_idx, :, 1]
    z_src = src[item_idx, :, 2]

    x_temp = temp[item_idx, :, 0]
    y_temp = temp[item_idx, :, 1]
    z_temp = temp[item_idx, :, 2]

    pred_t_matrix = preds[item_idx]

    ones_col = np.ones((x_src.shape[0], 1))
    ext_src = np.concatenate((src[item_idx], ones_col), axis=1)
    transformed = ext_src.dot(pred_t_matrix.T)
    transformed = transformed[:, :-1]

    x_new = transformed[:, 0]
    y_new = transformed[:, 1]
    z_new = transformed[:, 2]

    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    ax.scatter(x_src, z_src, y_src, alpha=0.35)
    ax.scatter(x_temp, z_temp, y_temp, alpha=0.35)
    ax.scatter(x_new, z_new, y_new, alpha=0.35)

    ax.set_axis_off()
    ax.set_xlim3d(limits[i][0], limits[i][1])
    ax.set_ylim3d(limits[i][0], limits[i][1])
    ax.set_zlim3d(limits[i][0], limits[i][1])
    ax.view_init(camera[i][0], camera[i][1])

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
# plt.show()

# Save the figure
fig_path = os.path.join(BASE_DIR, 'results', model_name + '.png')
plt.savefig(fig_path)

print('Image successfully saved to {}'.format(fig_path))
