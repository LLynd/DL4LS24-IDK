import wandb
from wandb.xgboost import WandbCallback
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc
from wandb import login

import numpy as np

from gradient_boosting_model import get_xgboost_data_and_model
from config_dc import Config
from dataloader import DataLoader
from visualizations import *
from utils import login_wandb


def evaluate_model(model, data, config: Config):
    pass

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
        )

    # get prediction
    pred = bst.predict(xg_test)
    accuracy = 

    wandb.summary["Accuracy"] = accuracy
    wandb.finish()
        