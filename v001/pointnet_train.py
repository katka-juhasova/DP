import argparse
import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
from v001.pointnet import pointnet_utils as utils
import v001.pointnet.pointnet_model as pointnet
from v001.pointnet.pointnet_generator import Generator


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=250,
                    help='Epoch to run [default: 250]')
args = parser.parse_args()


NUM_POINT = args.num_point
NUM_CLASS = 40
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(BASE_DIR, 'data', 'ModelNet40')
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)

# Load model
model = pointnet.get_model(num_point=NUM_POINT, num_class=NUM_CLASS)
# model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["sparse_categorical_accuracy"],
)

# Train with WandB callbacks
wandb_config = {
    'model': 'PointNet',
    'num_point': NUM_POINT,
    'num_class': NUM_CLASS,
    'batch_size': BATCH_SIZE,
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
wandb.run.name = 'PointNet-' + wandb.run.id

train_generator = Generator(train_files, num_point=NUM_POINT,
                            batch_size=BATCH_SIZE)
test_generator = Generator(test_files, num_point=NUM_POINT,
                           batch_size=BATCH_SIZE)

# Model checkpoint callback
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
checkpoint_dir = os.path.join(BASE_DIR, 'models',
                              timestamp + "_%s" % wandb.run.name)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(
        checkpoint_dir,
        (r"model.epoch{epoch:03d}-loss{loss:.2f}-"
         r"acc{sparse_categorical_accuracy:.2f}-"
         r"val_loss{val_loss:.2f}-"
         r"val_acc{val_sparse_categorical_accuracy:.2f}.h5'")
    ),
    monitor='val_sparse_categorical_accuracy',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='max')

# Finally fit
model.fit(train_generator,
          epochs=EPOCHS, validation_data=test_generator,
          callbacks=[WandbCallback(), checkpoint_callback], verbose=1)

run.finish()
