import init
from src.config_dc import Config
from src.inference import run_inference

import pandas as pd
import os
from datetime import datetime


config = Config(
    seed = 42,
    method = 'linear',
    infer = True,
    num_classes=14,
    wandb_project = 'DL4LS24-IDK',
    ann_data_path = 'data/test/cell_data.h5ad',
    img_data_path = 'data/test/images_masks',
    model_path = 'results/mlp_model.h5.pt',
    res_path = str(os.path.join('results', 'INFER_MLP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.csv'))
)

results = run_inference(config)
results.to_csv(config.res_path)