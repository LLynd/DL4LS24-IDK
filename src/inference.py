from src.config_dc import Config
from src.dataloader import CustomDataLoader
from src.prepare_output_df import new_predictions, results_to_df
from src.linear_model import get_model_for_inference

import pickle 
import torch
import xgboost as xgb


def load_model(path, method, inp_shape, config):
    if method == 'xgboost':
        model = xgb.XGBClassifier()
        model.load_model(path)
        
    elif method == 'logistic':
        with open(path, 'rb') as f:
            model = pickle.load(f)

    elif method == 'linear':
        model = get_model_for_inference(config, inp_shape)
        model.load_state_dict(torch.load(path))
        
    elif method == 'starling':
        pass
    
    return model

      
def run_inference(config):
    cdl = CustomDataLoader(config)
    X = cdl.get_data(preprocess=True)
    inp_shape = X.shape[1]
    model = load_model(config.model_path, config.method, inp_shape, config)
    if config.method == 'xgboost' or config.method == 'logistic': 
        results = model.predict(X)
        probabilities = model.predict_proba(X)
        results = results_to_df(results, probabilities, cdl.sample_ids, cdl.label_encoder)
    elif config.method == 'linear' or config.method == 'mlp':
        results = new_predictions(model, cdl.sample_ids, X, cdl.label_encoder)
    elif config.method == 'starling':
        #results = model.predict(X)
        pass
    
    return results