import os
import tensorflow as tf
import corsnet.corsnet as corsnet
import pointnet.pointnet as pointnet


NUM_POINTS = 1024
NUM_CLASSES = 40
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Plot CorsNet
model = corsnet.get_model(num_points=NUM_POINTS)
img_path = os.path.join(BASE_DIR, 'corsnet_architecture.png')
tf.keras.utils.plot_model(model, to_file=img_path, show_shapes=True)

# Plot PointNet
model = pointnet.get_model(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
img_path = os.path.join(BASE_DIR, 'pointnet_architecture.png')
tf.keras.utils.plot_model(model, to_file=img_path, show_shapes=True)

# img_path =
# tf.keras.utils.plot_model(model, to_file=img_path, show_shapes=True)