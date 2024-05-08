import os
import typing as t

from datetime import datetime
from dataclasses import asdict, dataclass, field, fields
from dataclasses_json import dataclass_json


# Dataclasses używamy do odpalania eksperymentów w mainie
# Przechowujemy tu wszystkie stałe, możecie je analogicznie dodawać i rozszerzać klasę
# inicjujecie ją np. tak: cfg = Config(n_epochs=100) - reszta wbije się na domyślne
# Oczywiście w trainie trzeba zrobić odniesienia do odpowiednich stałych z dc

@dataclass_json
@dataclass
class Config:
    seed: int = 42
    method: str = 'linear' #metoda jakiej uzywamy, możemy to też zmienić na podawanie obiektu klasyfikatora jesli tak wolicie
    wandb_project: str = 'DL4LS24-IDK'
    wandb_api_key: str = '...'
    
    num_epochs: int = 500
    learning_rate: float = 1e-6
    test_size: float = 0.2

    hidden_size: int = 100
    batch_size: int = 32
    
    #uncertainty analysis params
    run_uncertainty_analysis: bool = False
    num_samples_uncertainty: int = 100
    
    #xgboost params
    base_score: float=0.5
    colsample_bylevel: int=1
    colsample_bytree: int=1
    max_depth: int=3
    min_child_weight: int=1
    gamma: float=0.0 
    
    dropout: bool = False
    
    #paths
    res_path: str = str(os.path.join('results', 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.npy')) #zmienic na destination path czy cos
    cfg_path: str = str(os.path.join('data', 'configs', 'CFG_EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.json'))
    
    ann_data_path: str = 'cell_data.h5ad'
    img_data_path: str = os.path.join('data', 'train', 'images_masks')
    
    run_name: str = 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))