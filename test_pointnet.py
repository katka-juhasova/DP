import os
import tensorflow as tf
from tensorflow import keras
import pointnet.utils as utils
import pointnet.pointnet as pointnet
from pointnet.generator import Generator
import matplotlib.pyplot as plt


NUM_POINTS = 1024
NUM_CLASSES = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.001

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

test_files = utils.get_data_files(TEST_FILES)
shapes_file = os.path.join(BASE_DIR,
                           'data/modelnet40_ply_hdf5_2048/shape_names.txt')

# Read shape names
with open(shapes_file) as f:
    shapes = f.read().splitlines()

# Test data generator
test_generator = Generator(test_files, num_point=NUM_POINTS,
                           batch_size=BATCH_SIZE)

# Load model
model = pointnet.get_model(num_points=NUM_POINTS, num_classes=NUM_CLASSES)

# Load weights
weights_path = os.path.join(
    BASE_DIR,
    (r"models/2021-10-27_07:38:35_PointNet-1zy4zmyd/"
     r"model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5")
)
model.load_weights(weights_path)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["sparse_categorical_accuracy"],
)

# Quantitative evaluation
# NOTE: Original PointNet accuracy: 89.2%
val_loss, val_acc = model.evaluate(test_generator, verbose=0)

# Qualitative evaluation
data, labels = test_generator.__getitem__(0)
pred = model.predict(data)
pred = tf.math.argmax(pred, -1)

# Plot points with predicted class and label
fig = plt.figure(figsize=(15, 20))

for i in range(BATCH_SIZE):
    ax = fig.add_subplot(4, 4, i + 1, projection="3d")
    ax.scatter(data[i, :, 0], data[i, :, 1], data[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            shapes[pred[i].numpy()], shapes[labels[i]]
        )
    )
    ax.set_axis_off()

# Save the figure
plt.suptitle('val_loss: {}, val_acc: {}'.format(val_loss, val_acc * 100))
fig_path = os.path.join(BASE_DIR, 'results/PointNet-1zy4zmyd.png')
plt.savefig(fig_path)
