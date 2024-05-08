from config_dc import Config
from experiment import run_experiment
import wandb

config = Config(
    seed = 42,
    method = 'logistic',
    wandb_api_key = "",
    num_epochs=10,
    learning_rate=1e-6,
    batch_size=100,
    hidden_size=100,
    num_samples_uncertainty=100,
    ann_data_path = 'data/train/cell_data.h5ad',
    img_data_path = 'data/train/images_masks'
                )

wandb.login()
results = run_experiment(config)
