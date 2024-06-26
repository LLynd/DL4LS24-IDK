from src.dataloader import CustomDataLoader

def get_mlp_data_and_model(config):
    dl = CustomDataLoader(config)
    data = dl.load_mlp(preprocess=True, 
                       split=config.test_size)
    
    model = ...
    
    return data, model

def train_mlp(model, dataloader, criterion, optimizer, num_epochs):
    pass

def predict_mlp(model, X_test, y_test):
    pass