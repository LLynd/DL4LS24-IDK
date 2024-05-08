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
    X_train, Y_train = CustomDataLoader.get_data(config)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train, dtype=torch.long)  

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearClassifier(input_size=X_tensor.shape[1], hidden_size=100, num_classes=14)
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    return X_tensor, Y_tensor, model, criterion, optimizer