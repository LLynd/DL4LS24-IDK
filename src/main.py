from config_dc import Config
from experiment import run_experiment

if __name__ == '__main__':
    config = Config(wandb_user = "...",
                    )
    results = run_experiment(config)
