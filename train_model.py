import argparse
import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.framework.ops import disable_eager_execution
import wandb
from wandb.keras import WandbCallback
import corsnet.model as corsnet
from corsnet.generator import Generator
from corsnet.losses import CorsNetLoss1
import corsnet.metrics as metrics


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
disable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Initial learning rate [default: 0.0001]')
parser.add_argument('--epochs', type=int, default=150,
                    help='Epoch to run [default: 150]')
parser.add_argument('--jitter', type=bool, default=True,
                    help='Use jitter for augmentation [default: True]')
parser.add_argument('--shuffle_points', type=bool, default=True,
                    help='Shuffle points in point clouds for augmentation \
                    [default: True]')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Data dir containing train & test dataset')
args = parser.parse_args()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_POINT = args.num_point
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
JITTER = args.jitter
SHUFFLE_POINTS = args.shuffle_points
DATA_DIR = os.path.join(BASE_DIR, args.data_dir)
DATA_BASE = os.path.basename(DATA_DIR.strip('/\\'))
SAME_POINTS = bool(int(DATA_BASE.split('_')[2]))
R_MIN = DATA_BASE.split('_')[4]
R_MAX = DATA_BASE.split('_')[5]
T_MIN = DATA_BASE.split('_')[7]
T_MAX = DATA_BASE.split('_')[8]


train_file = os.path.join(DATA_DIR, 'train_dataset.h5')
test_file = os.path.join(DATA_DIR, 'test_dataset.h5')

train_generator = Generator(train_file, batch_size=BATCH_SIZE,
                            jitter=JITTER, shuffle_points=SHUFFLE_POINTS)
test_generator = Generator(test_file, batch_size=BATCH_SIZE, jitter=False,
                           shuffle=False, shuffle_points=False)
model = corsnet.get_model(num_point=NUM_POINT)
# model.summary()

model.compile(
    loss=CorsNetLoss1(),
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mse", "acc", metrics.rmse_r, metrics.rmse_t]
)

# Train with WandB callbacks
wandb_config = {
    'model': 'CorsNet_v2',
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
wandb.run.name = 'CorsNet_v2-' + wandb.run.id

# Learning rate callback
# lr_callback = tf.keras.callbacks.LearningRateScheduler(corsnet.lr_scheduler)

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
model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator,
          # callbacks=[WandbCallback(), checkpoint_callback, lr_callback],
          callbacks=[WandbCallback(), checkpoint_callback],
          verbose=1)

run.finish()
