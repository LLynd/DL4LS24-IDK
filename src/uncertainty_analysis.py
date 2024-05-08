import numpy as np
import matplotlib.pyplot as plt

import torch

from src.visualizations import visualize_uncertainties


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
    model.dropout2.p = 0.3
    
    uncertainties = uncertainty_analysis(model, dataloader, num_samples=num_samples)
    uncertainties = np.nan_to_num(uncertainties)
    visualize_uncertainties(uncertainties)
    return uncertainties


model_dropout = LinearClassifierWithDropout(input_size=X_tensor.shape[1], hidden_size=100, num_classes=14, dropout_prob=0.01)
criterion_drop = nn.CrossEntropyLoss()  # Cross-entropy loss for classification  task
optimizer_drop = optim.Adam(model_dropout.parameters(), lr=0.003)
dataloader_drop = DataLoader(dataset, batch_size=32, shuffle=True)


# We train the model with dropout_prob=0
train(model_dropout, dataloader_drop, criterion_drop, optimizer_drop, num_epochs=10)
