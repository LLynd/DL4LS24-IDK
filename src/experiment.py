import numpy as np
import wandb
import os
import pickle

from src.gradient_boosting_model import get_xgboost_data_and_model
from src.mlp_model import get_mlp_data_and_model, train_mlp, predict_mlp
from src.linear_model import get_linear_data_and_model, train_linear, predict_linear
from src.config_dc import Config
from src.dataloader import CustomDataLoader
from src.visualizations import *
from src.evaluate_model import evaluate_model
from src.uncertainty_analysis import run_uncertainty_analysis
from src.logistic_regression_model import get_logistic_data_and_model


def run_experiment(config: Config):
    wandb.login()
    
    if config.method == 'xgboost':
        data, model = get_xgboost_data_and_model(config)
        
        params = {
            "method": config.method,
            "learning_rate": config.learning_rate,
            "base_score": config.base_score,
            "colsample_bylevel": config.colsample_bylevel,
            "colsample_bytree": config.colsample_bytree,
            "max_depth": config.max_depth,
            "min_child_weight": config.min_child_weight,
            "gamma": config.gamma,
            "test_size": config.test_size,
        }

        run = wandb.init(
            project=config.wandb_project,
            config=params,
            name=config.run_name
        )
        
        if config.test_size is None:
            X_train, y_train = data[0], data[1]
        else:
            X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        
        model.fit(X_train, y_train)
        
        if config.test_size is not None:
            y_pred = model.predict(X_test)
            plot = bar_plot_accuracy_per_class(y_pred, y_test, 'Accuracy per class xgboost')
            #wandb.log({"Accuracy per class xgboost": plot})
            y_pred_proba = model.predict_proba(X_test)
        
    elif config.method == 'logistic':
        data, model = get_logistic_data_and_model(config)
        params = {
            "method": config.method,
            "learning_rate": config.learning_rate,
            "test_size": config.test_size,
        }

        run = wandb.init(
            project=config.wandb_project,
            config=params,
            name=config.run_name
        )
        
        if config.test_size is None:
            X_train, y_train = data[0], data[1]
        else:
            X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        
        model.fit(X_train, y_train)
        
        if config.test_size is not None:
            y_pred = model.predict(X_test)
            plot = bar_plot_accuracy_per_class(y_pred, y_test, 'Accuracy per class Logistic Regresion')
            #wandb.log({"Accuracy per class Logistic Regresion": plot})
            y_pred_proba = model.predict_proba(X_test)
        
    elif config.method == 'mlp':
        data, dataloader, model, criterion, optimizer = get_mlp_data_and_model(config)
        
        params = {
            "method": config.method,
            "learning_rate": config.learning_rate,
            "test_size": config.test_size,
        }

        run = wandb.init(
            project=config.wandb_project,
            config=params,
            name=config.run_name
        )
        
        if config.test_size is None:
            X_train, y_train = data[0], data[1]
        else:
            X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]
        
        model = train_mlp(model, dataloader, criterion, optimizer, num_epochs=config.num_epochs, config=config)

        if config.test_size is not None:
            y_pred, y_pred_proba = predict_mlp(model, X_test, y_test)
        
        
    elif config.method == 'linear':
        data, dataloader, model, criterion, optimizer = get_linear_data_and_model(config)
        
        params = {
            "method": config.method,
            "learning_rate": config.learning_rate,
            "test_size": config.test_size,
        }

        run = wandb.init(
            project=config.wandb_project,
            config=params,
            name=config.run_name
        )
        
        if config.test_size is None:
            X_train, y_train = data[0], data[1]
        else:
            X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]

        model = train_linear(model, dataloader, criterion, optimizer, num_epochs=config.num_epochs)

        if config.test_size is not None:
            y_pred, y_pred_proba = predict_linear(model, X_test, y_test)
            plot = bar_plot_accuracy_per_class(y_pred, y_test, 'Accuracy per class MLP')
            #wandb.log({"Accuracy per class MLP": plot})
        
    elif config.method == 'starling':
        pass
    
    elif config.method == 'cnn':
        pass
        
    if config.test_size is not None:
        accuracy, macro_f1, auc, average_precision = evaluate_model(y_pred, y_pred_proba, y_test)
        
        run.define_metric("Accuracy", summary="max")
        run.define_metric("Macro F1", summary="max")
        run.define_metric("AUC (OvR)")
        run.define_metric("Average Precision")
        
        run.summary["Accuracy"] = accuracy
        run.summary["Macro F1"] = macro_f1
        run.summary["AUC (OvR)"] = auc
        run.summary["Average Precision"] = average_precision
    
    if config.run_uncertainty_analysis is True:
        uncertainties = run_uncertainty_analysis(model, dataloader, config.num_samples_uncertainty)

    #run.log_model(path=r'results/model.h5', name='model')
    if config.method == 'linear':
        torch.save(model.state_dict(), config.model_path+'.pt')
    elif config.method == 'xgboost':
        model.save_model(config.model_path+'.bst')
    elif config.method == 'starling':
        pass
    elif config.method == 'logistic':
        with open(config.model_path+'.pkl', 'wb') as f:
            pickle.dump(model, f)
     
    wandb.finish()
