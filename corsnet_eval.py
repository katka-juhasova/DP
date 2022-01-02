import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'corsnet'))
import corsnet_utils as utils
import corsnet_model as corsnet
from corsnet_generator import Generator
from corsnet_loss1 import CorsNetLoss1
import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True,
                    help='Trained weights path')
parser.add_argument('--num_point', type=int, default=1024,
                    help='Point number [256/512/1024] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training [default: 32]')
args = parser.parse_args()


WEIGHTS = args.weights
NUM_POINT = args.num_point
BATCH_SIZE = args.batch_size


# CorsNet-cd5u6xe5-v5
# MODEL_DIR = "2021-12-20_00:24:08_CorsNet-cd5u6xe5"
# WEIGHTS = (r"model.epoch156-loss1.04-acc0.95-mse0.00"
#            r"-val_loss1.05-val_acc0.96-val_mse0.00.h5")

# CorsNet-2kp51w3l-v4
# MODEL_DIR = "2021-12-16_20:35:53_CorsNet-2kp51w3l"
# WEIGHTS = (r"model.epoch178-loss1.05-acc0.96-mse0.00"
#            r"-val_loss1.07-val_acc0.96-val_mse0.00.h5")

# CorsNet-23rj48af-v3
# MODEL_DIR = "2021-12-15_09:56:37_CorsNet-23rj48af"
# WEIGHTS = (r"model.epoch191-loss1.02-acc0.96-mse0.00"
#            r"-val_loss1.03-val_acc0.97-val_mse0.00.h5")

# CorsNet-wbdc07oz-v2
# MODEL_DIR = "2021-12-14_22:32:12_CorsNet-wbdc07oz"
# WEIGHTS = (r"model.epoch060-loss4.18-acc0.95-mse0.01"
#            r"-val_loss4.68-val_acc0.86-val_mse0.04.h5")

# CorsNet-32j1ocp4-v1
# MODEL_DIR = "2021-12-13_23:49:00_CorsNet-32j1ocp4"
# WEIGHTS = (r"model.epoch183-loss1.05-acc0.96-mse0.00"
#            r"-val_loss1.06-val_acc0.96-val_mse0.00.h5")


DATA_DIR = os.path.join(BASE_DIR, 'data', 'CorsNet_eval')
TEST_FILES = os.path.join(DATA_DIR, 'test_files.txt')

test_files = utils.get_data_files(TEST_FILES)

# Test data generator
test_generator = Generator(test_files,
                           batch_size=BATCH_SIZE,
                           jitter=False,
                           shuffle=False,
                           shuffle_points=False)

model_name = WEIGHTS.split('_')[-1].split('/')[0].split('_')[-1]

# Load model and weights
# CorsNet-23rj48af-v3
weights_path = os.path.join(BASE_DIR, 'models', WEIGHTS)

model = corsnet.get_model(num_point=NUM_POINT)
model.load_weights(weights_path)

model.compile(
    loss=CorsNetLoss1(),
    metrics=["mse", "acc", metrics.rmse_r, metrics.rmse_t]
)

# Quantitative evaluation
loss, mse, acc, rmse_r, rmse_t = model.evaluate(test_generator, verbose=0)

print(model_name)
print('acc: {:.5f}, loss: {:.5f}, MSE: {:.5f}, RMSE_R: {:.5f}, RMSE_t: {:.5f}'.
      format(acc, loss, mse, rmse_r, rmse_t))
