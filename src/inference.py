from config_dc import Config

def load_model(path):
    pass

def run_inference(config):
    X = load_data(config)
    model = load_model(config.saved_model_path)
    model.predict