import os
from typing import Dict, List
import pandas as pd

DATA_DIR = 'data'

class Data:

    def __init__(self):
        self.get: Dict[str, pd.DataFrame] = {}

def get_data(filename: str) -> pd.DataFrame:
    file = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(file)
    return df

def get_all_data() -> Data:
    files = ['train', 'dev_in', 'dev_out']
    data = Data()
    for file in files:
        data.get[file] = get_data(file+".csv")
    return data