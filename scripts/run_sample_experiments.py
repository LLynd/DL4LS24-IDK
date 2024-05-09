from src.config_dc import Config
from src.experiment import run_experiment
import wandb

config = Config(
    seed = 42,
    method = 'xgboost',
    wandb_api_key = "",
    num_epochs=10,
    learning_rate=1e-6,
    batch_size=100,
    hidden_size=100,
    num_samples_uncertainty=100,
    ann_data_path = 'C:/Users/Dell/Documents/Deep_learning_for_life_science/Projekt_zaliczeniowy/Repo_v3_z_gitem/DL4LS24-IDK/data/train/cell_data.h5ad',
    img_data_path = 'C:/Users/Dell/Documents/Deep_learning_for_life_science/Projekt_zaliczeniowy/Repo_v3_z_gitem/DL4LS24-IDK/data/train/images_masks'
                )

results = run_experiment(config)