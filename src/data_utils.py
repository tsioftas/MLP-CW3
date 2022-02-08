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
        df = get_data(file+".csv")
        data.get[file+"_y"] = df['fact_temperature']
        df = df.drop(df.filter(regex='fact_*').columns)
        df = df.drop("climate")
        data.get[file+"_x"] = df
    return data