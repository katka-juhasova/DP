# DP

## Download & Installation

1. Download and extract data from https://drive.google.com/file/d/1zvu8rQTbBEtBUpoYVM9TBJUnzQpNlmMY/view?usp=sharing to repository root.
2. Download and extract pretrained models from https://drive.google.com/file/d/1HZly035M8XkMTMUJKdQvBN9IJ1YydBVe/view?usp=sharing to repository root. 
3. Install required packages from requirements.txt.

## Usage

Data preprocessing:
```
python3 data_preprocess.py
```

PointNet training:
```
python3 pointnet_train.py
```

PointNet quantitative evaluation:
```
python3 pointnet_eval.py --weights "2021-10-27_07:38:35_PointNet-1zy4zmyd/model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5"
```

PointNet qualitative evaluation:
```
python3 tmp/pointnet_eval_img.py --weights "2021-10-27_07:38:35_PointNet-1zy4zmyd/model.epoch239-loss2.17-acc0.95-val_loss2.63-val_acc0.87.h5"
```

CorsNet training:
```
python3 corsnet_train.py 
```

CorsNet quantitative evaluation:
```
python3 corsnet_eval.py --weights "2021-12-15_09:56:37_CorsNet-23rj48af/model.epoch191-loss1.02-acc0.96-mse0.00-val_loss1.03-val_acc0.97-val_mse0.00.h5"
```

CorstNet qualitative evaluation:
```
python3 tmp/corsnet_eval_img.py --weights "2021-12-15_09:56:37_CorsNet-23rj48af/model.epoch191-loss1.02-acc0.96-mse0.00-val_loss1.03-val_acc0.97-val_mse0.00.h5"
```
