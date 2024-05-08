import numpy as np
import wandb

from wandb.xgboost import WandbCallback
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc
from wandb import login

from src.gradient_boosting_model import get_xgboost_data_and_model
from src.mlp_model import get_mlp_data_and_model
from src.linear_model import get_linear_data_and_model
from src.config_dc import Config
from src.dataloader import CustomDataLoader
from src.visualizations import *
from src.evaluate_model import evaluate_model
from src.uncertainty_analysis import run_uncertainty_analysis
    

def run_experiment(config: Config):
    login(key=...,
          )
    
    if config.method == 'xgboost':
        data, model = get_xgboost_data_and_model(config)
        train, test = data[0], data[1]
        
        # setup parameters for xgboost
        param = {
            "objective" : "multi:softmax",
            "eta" : 0.1,
            "max_depth": 6,
            "nthread" : 4,
            "num_class" : 6
        }

        wandb.init(
            project=config.wandb_project,
            # track hyperparameters and run metadata
            config=param
        )

        watchlist = [(xg_train, "train"), (xg_test, "test")]

        # pass WandbCallback to the booster to log its configs and metrics
        bst = xgb.train(
            param, xg_train, num_round, evals=watchlist,
            callbacks=[WandbCallback()]
        )pred = bst.predict(xg_test)
    
    elif config.method == 'mlp':
        ...
        
    elif config.method == 'linear':
        ...

    accuracy, macro_f1, auc, average_precision = evaluate_model(y_pred, y_pred_proba, y_test)
    
    wandb.summary["Accuracy"] = accuracy
    wandb.summary["Macro F1"] = macro_f1
    
    if config.run_uncertainty_analysis is True:
        uncertainties = run_uncertainty_analysis(model, dataloader, config.num_samples_uncertainty)

    wandb.finish()
        