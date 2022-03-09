import os
from typing import Dict, List
import pandas as pd
import numpy as np

import torch

DATA_DIR = 'data'

class Data:

    def __init__(self):
        self.get: Dict[str, pd.DataFrame] = {}

def get_data(filename: str) -> pd.DataFrame:
    file = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(file)
    return df

def get_all_data() -> Data:
    #files = ['shifts_canonical_train', 'shifts_canonical_dev_in', 'shifts_canonical_dev_out', 'shifts_canonical_eval_in', 'shifts_canonical_eval_out']
    files = ['shifts_canonical_dev_in', 'shifts_canonical_dev_out', 'shifts_canonical_eval_in']
    data = Data()
    for file in files:
        df = get_data(file+".csv")
        df.fillna(value=-1, inplace=True)
        data.get[file+"_y"] = df['fact_temperature']
        df = df[df.columns.drop(['climate'] + list(df.filter(regex='fact_')))].astype(np.float32)
        data.get[file+"_x"] = df
    return data

