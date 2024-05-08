import xgboost as xgb

from src.dataloader import CustomDataLoader

def get_xgboost_data_and_model(config):
    dl = CustomDataLoader(config)
    data = dl.load_xgboost(preprocess=True, 
                           split=config.test_size)
    
    model = xgb.XGBClassifier(base_score=config.base_score, 
                              colsample_bylevel=config.colsample_bylevel, 
                              colsample_bytree=config.colsample_bytree,
                              max_depth=config.max_depth,
                              min_child_weight=config.min_child_weight,
                              gamma=config.gamma, 
                              learning_rate=config.learning_rate, 
                              seed=config.seed)
    
    return data, model

