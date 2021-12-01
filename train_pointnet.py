import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import pointnet.utils as utils
import pointnet.pointnet as pointnet
from pointnet.generator import Generator
import wandb
from wandb.keras import WandbCallback

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

NUM_POINTS = 1024
NUM_CLASSES = 40
BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 0.001

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')
TRAIN_FILES = os.path.join(DATA_DIR, 'train_files.txt')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

train_files = utils.get_data_files(TRAIN_FILES)
test_files = utils.get_data_files(TEST_FILES)

# Load model
model = pointnet.get_model(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
# model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["sparse_categorical_accuracy"],
)

# Train with WandB callbacks
wandb_config = {
    'model': 'PointNet',
    'num_points': NUM_POINTS,
    'num_classes': NUM_CLASSES,
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

train_generator = Generator(train_files, num_point=NUM_POINTS,
                            batch_size=BATCH_SIZE)
test_generator = Generator(test_files, num_point=NUM_POINTS,
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
