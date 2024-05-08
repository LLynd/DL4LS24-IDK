from sklearn.linear_model import LogisticRegression

from dataloader import CustomDataLoader

def get_logistic_data_and_model(config):
    dl = CustomDataLoader(config)
    data = dl.get_data(preprocess=True)
    model = LogisticRegression(multi_class='multinomial', 
                                       solver='lbfgs', 
                                       max_iter=1000)
    
    return data, model
