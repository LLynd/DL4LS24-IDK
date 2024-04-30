import xgboost as xgb

from dataloader import DataLoader

def get_xgboost_data_and_model(config):
    dl = DataLoader(config)
    data = dl.load_xgboost(preprocess=True, split=config.split)
    
    model = 
    
    return data, model