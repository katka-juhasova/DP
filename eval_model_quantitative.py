import argparse
import os
import corsnet.model as corsnet
from corsnet.generator import Generator
from corsnet.losses import CorsNetLoss1
import corsnet.metrics as metrics


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='Data dir containing train & test dataset')
parser.add_argument('--weights', type=str, required=True,
                    help='Trained weights path')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for evaluation [default: 32]')
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

model.compile(
    loss=CorsNetLoss1(),
    metrics=["mse", "acc", metrics.rmse_r, metrics.rmse_t]
)

# Quantitative evaluation
loss, mse, acc, rmse_r, rmse_t = model.evaluate(test_generator, verbose=0)

print(model_name)
print('acc: {:.5f}, loss: {:.5f}, MSE: {:.5f}, RMSE_R: {:.5f}, RMSE_t: {:.5f}'.
      format(acc, loss, mse, rmse_r, rmse_t))
