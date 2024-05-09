import xgboost as xgb
import torch

from src.dataloader import CustomDataLoader

def get_xgboost_data_and_model(config):
    dl = CustomDataLoader(config)
    data = dl.get_data(preprocess=True)
    model = xgb.XGBClassifier(base_score=config.base_score, 
                              colsample_bylevel=config.colsample_bylevel, 
                              colsample_bytree=config.colsample_bytree,
                              max_depth=config.max_depth,
                              min_child_weight=config.min_child_weight,
                              gamma=config.gamma, 
                              learning_rate=config.learning_rate, 
                              seed=config.seed,
                              device="cuda" if torch.cuda.is_available() else "cpu")
    
    return data, model
