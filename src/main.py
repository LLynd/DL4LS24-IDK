from config_dc import Config
from experiment import run_experiment

if __name__ == '__main__':
    config = Config(
        seed = 42,
        method = 'linear',
        wandb_project="...",
        wandb_user = "...",
        n_epochs=100,
        learning_rate=1e-6,
        batch_size=100,
        hidden_size=100,
        num_samples_uncertainty=100,
        ann_data_path = 'data/train/cell_data.h5ad',
        img_data_path = 'data/train/images_masks'
                    )
    
    results = run_experiment(config)
