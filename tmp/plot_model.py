import os
import tensorflow as tf
import pointnet.pointnet as pointnet


NUM_POINTS = 1024
NUM_CLASSES = 40
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = pointnet.get_model(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
img_path = os.path.join(BASE_DIR, 'pointnet_architecture.png')
tf.keras.utils.plot_model(model, to_file=img_path, show_shapes=True)
