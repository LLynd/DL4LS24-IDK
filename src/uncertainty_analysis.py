import numpy as np
import matplotlib.pyplot as plt

import torch

from visualizations import visualize_uncertainties


def uncertainty_analysis(model, dataloader, num_samples=100):
    # Works for: linear model, linear model with dropout
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            mc_predictions = torch.stack([model(inputs).softmax(dim=1) for _ in range(num_samples)], dim=0)
            mean_prediction = mc_predictions.mean(dim=0)
            predictive_entropy = -(mean_prediction * torch.log(mean_prediction)).sum(dim=1)
            uncertainties.append(predictive_entropy.numpy())
    return np.concatenate(uncertainties)

def run_uncertainty_analysis(model, dataloader, num_samples=100):
    model.dropout1.p = 0.3 #universalize this
    model.dropout2.p = 0.3 #universalize this
    
    uncertainties = uncertainty_analysis(model, dataloader, num_samples=num_samples)
    uncertainties = np.nan_to_num(uncertainties)
    visualize_uncertainties(uncertainties)
    return uncertainties
