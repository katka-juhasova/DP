import argparse
import os
import sys
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.framework.ops import disable_eager_execution
import wandb
from wandb.keras import WandbCallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'corsnet'))
import corsnet_utils as utils
import corsnet_model as corsnet
from corsnet_generator import Generator
from corsnet_loss1 import CorsNetLoss1


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
disable_eager_execution()

# Original CorsNet settings:
# batch_size = 32
# learning_rate = 0.0001
# epochs = 300
parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.00005,
                    help='Initial learning rate [default: 0.00005]')
parser.add_argument('--epochs', type=int, default=200,
                    help='Epoch to run [default: 200]')
parser.add_argument('--pointnet_weights', type=str, default=None,
                    help='Path to pretrained PointNet weights [default: None]')
parser.add_argument('--pointnet_train', type=bool, default=True,
                    help='If True, set PointNet weights as trainable \
                    [default: True]')
parser.add_argument('--jitter', type=bool, default=True,
                    help='Use jitter for augmentation [default: True]')
parser.add_argument('--shuffle_points', type=bool, default=True,
                    help='Shuffle points in point clouds for augmentation \
                    [default: True]')
parser.add_argument('--same_points', type=bool, default=False,
                    help='Setting for src and temp point cloud sampling. \
                    If True, both point clouds are consist of the same points \
                    [default: False]')
parser.add_argument('--r_min', type=float, default=0.0,
                    help='Lower limit for rotation in degrees [default: 0.0]')
parser.add_argument('--r_max', type=float, default=45.0,
                    help='Upper limit for rotation in degrees [default: 45.0]')
parser.add_argument('--t_min', type=float, default=0.0,
                    help='Lower limit for translation [default: 0.0]')
parser.add_argument('--t_max', type=float, default=0.8,
                    help='Upper limit for translation [default: 0.8]')
args = parser.parse_args()


NUM_POINT = args.num_point
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
POINTNET_WEIGHTS = args.pointnet_weights
POINTNET_TRAIN = args.pointnet_train
JITTER = args.jitter
SHUFFLE_POINTS = args.shuffle_points
SAME_POINTS = args.same_points
R_MIN = args.r_min
R_MAX = args.r_max
T_MIN = args.t_min
T_MAX = args.t_max


DATA_DIR = os.path.join(BASE_DIR, 'data', 'CorsNet')
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)

train_generator = Generator(train_files, batch_size=BATCH_SIZE,
                            jitter=JITTER, shuffle_points=SHUFFLE_POINTS)
test_generator = Generator(test_files, batch_size=BATCH_SIZE,
                           jitter=JITTER, shuffle_points=SHUFFLE_POINTS)

# Load model with pre-trained PointNet weights
# best weights = 2021-10-27_07:38:35_PointNet-1zy4zmyd/
# model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5
weights = (os.path.join(BASE_DIR, 'models', POINTNET_WEIGHTS)
           if POINTNET_WEIGHTS else None)

model = corsnet.get_model(num_point=NUM_POINT, pointnet_weights=weights,
                          pointnet_trainable=POINTNET_TRAIN)
# model.summary()

model.compile(
    loss=CorsNetLoss1(),
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mse", "acc"]
)

# Train with WandB callbacks
wandb_config = {
    'model': 'CorsNet',
    # Preprocessing config
    'same_points': SAME_POINTS,
    'r_min': R_MIN,
    'r_max': R_MAX,
    't_min': T_MIN,
    't_max': T_MAX,
    # Generator config
    'num_point': NUM_POINT,
    'batch_size': BATCH_SIZE,
    'jitter': JITTER,
    'shuffle_points': SHUFFLE_POINTS,
    # Training config
    'pointnet_weights': POINTNET_WEIGHTS,
    'pointnet_trainable': POINTNET_TRAIN,
    'optimizer': 'Adam',
    'learning_rate': LEARNING_RATE
}

# WandB init
run = wandb.init(
    project='CorsNet',
    reinit=True,
    config=wandb_config
)

# Set the run name to the run ID
wandb.run.name = 'CorsNet-' + wandb.run.id

# Learning rate callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(corsnet.lr_scheduler)

# Model checkpoint callback
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
checkpoint_dir = os.path.join(BASE_DIR, 'models',
                              timestamp + "_%s" % wandb.run.name)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(
        checkpoint_dir,
        (r"model.epoch{epoch:03d}-loss{loss:.2f}-acc{acc:.2f}-mse{mse:.2f}-"
         r"val_loss{val_loss:.2f}-val_acc{val_acc:.2f}-val_mse{val_mse:.2f}.h5"
         )),
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='min')

# Finally fit
model.fit(train_generator,
          epochs=EPOCHS, validation_data=test_generator,
          callbacks=[WandbCallback(), checkpoint_callback, lr_callback],
          verbose=1)

run.finish()
