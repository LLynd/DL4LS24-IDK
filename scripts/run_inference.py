import init
from src.config_dc import Config
from src.inference import run_inference

import pandas as pd
import os
import datetime


config = Config(
    seed = 42,
    method = 'xgboost',
    infer = True,
    num_classes=14,
    wandb_project = 'DL4LS24-IDK',
    ann_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\cell_data.h5ad',
    img_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\images_masks',
    model_path = r'',
    res_path = str(os.path.join('results', 'INFER_XGBOOST_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.csv'))
)

results = run_inference(config)
results.to_csv(config.res_path)

config = Config(
    seed = 42,
    method = 'linear',
    infer = True,
    wandb_project = 'DL4LS24-IDK',
    ann_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\cell_data.h5ad',
    img_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\images_masks',
    model_path = r'',
    res_path = str(os.path.join('results', 'INFER_LINEARMLP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.csv'))
)

results = run_inference(config)
results.to_csv(config.res_path)

config = Config(
    seed = 42,
    method = 'logistic',
    infer = True,
    wandb_project = 'DL4LS24-IDK',
    ann_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\cell_data.h5ad',
    img_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\images_masks',
    model_path = r'',
    res_path = str(os.path.join('results', 'INFER_LOGISTIC_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.csv'))
)

results = run_inference(config)
results.to_csv(config.res_path)

""" config = Config(
    seed = 42,
    method = 'starling',
    infer = True,
    wandb_project = 'DL4LS24-IDK',
    ann_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\cell_data.h5ad',
    img_data_path = r'C:\Users\lniedzwiedzki\Desktop\DL4LS24-IDK\data\test\images_masks',
    model_path = r'',
    res_path = str(os.path.join('results', 'INFER_STARLING_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.csv'))
)

results = run_inference(config)
results.to_csv(config.res_path) """