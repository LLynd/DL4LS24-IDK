import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from src.dataloader import CustomDataLoader


class LinearClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)  
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
class LinearClassifierWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.01):
        super(LinearClassifierWithDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout with probability dropout_prob
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout with probability dropout_prob
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout during forward pass
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout during forward pass
        x = self.fc3(x)
        return x

def get_linear_data_and_model(config):
    cdl = CustomDataLoader(config)
    if config.test_size is not None:
        X_train, Y_train, X_test, Y_test = cdl.get_data(preprocess=True)
    elif config.test_size is None:
        X_train, Y_train = cdl.get_data(preprocess=True)
    
    num_classes = len(set(Y_train))
    config.num_classes = num_classes
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.long)  

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    if config.dropout:
        model = LinearClassifierWithDropout(input_size=X_tensor.shape[1], 
                                            hidden_size=config.hidden_size, 
                                            num_classes=num_classes)
    elif not config.dropout:
        model = LinearClassifier(input_size=X_tensor.shape[1],
                                hidden_size=config.hidden_size, 
                                num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), 
                           lr=config.learning_rate,)

    if config.test_size is not None:
        data = (X_train, Y_train, X_test, Y_test)
    elif config.test_size is None:
        data = (X_train, Y_train) #test size?
        
    return data, dataloader, model, criterion, optimizer

def get_model_for_inference(config, inp_shape):
    if config.dropout:
        model = LinearClassifierWithDropout(input_size=inp_shape, 
                                            hidden_size=config.hidden_size, 
                                            num_classes=config.num_classes)
    elif not config.dropout:
        model = LinearClassifier(input_size=inp_shape,
                                hidden_size=config.hidden_size, 
                                num_classes=config.num_classes)

    return model
def train_linear(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate loss and acc
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    
    return model

def predict_linear(model, X_test, Y_test):
    X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    Y_tensor_test = torch.tensor(Y_test, dtype=torch.long)  

    with torch.no_grad():
        outputs = model(X_tensor_test)
        print(outputs.shape)
        predictions_probabilities = torch.softmax(outputs,axis=1)
        #predictions_probabilities /= predictions_probabilities.sum(axis=1, keepdims=True)
        predictions_probabilities = predictions_probabilities.detach().numpy() 
        predictions = np.argmax(predictions_probabilities, axis=1)
    
    return predictions, predictions_probabilities