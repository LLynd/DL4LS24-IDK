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
    wandb_project: str = '...'
    wandb_user: str = '...'
    n_epochs: int = 500
    col_batch_size: int = 100
    learning_rate: float = 1e-6
    
    res_path: str = str(os.path.join('results', 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.npy')) #zmienic na destination path czy cos
    cfg_path: str = str(os.path.join('data', 'configs', 'CFG_EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'.json'))
    plot_history_path: str = str(os.path.join('results', 'EXP_'+str(datetime.today().strftime('%Y-%m-%d_%H:%M:%S'))+'_history.png'))
    
    anndaata_path: str = 'cell_data.h5ad'
    imgdata_path: str = os.path.join('data', 'train', 'images_masks')