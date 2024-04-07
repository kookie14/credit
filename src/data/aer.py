import logging
import pandas as pd
from data.base_data import Dataset
import sys

class AERDataset(Dataset):
    def __init__(self, data_path: str, class_column_name: str = 'class'):
        super().__init__(data_path, class_column_name)
        logging.info(f"data_path = {self.data_path}")
        self.raw_data = pd.read_csv(data_path, skiprows=0, sep=",")
        logging.info(f"raw_data shape = {self.raw_data.shape}")
        self._set_column_names()
        self._remove_missing_values()
    
    def _set_column_names(self):
        self.raw_data.columns = [self.class_column_name] + ["A"+ str(i) for i in range(1, 12)] 

    def _remove_missing_values(self):
        logging.info(f"has NaNs = {self.raw_data.isnull().values.any()}")


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    aer_data = AERDataset(
        data_path='/home/cuctt/credit/data/AER/AER_credit_card_data.csv', 
        class_column_name='class'
    )
    print(aer_data.raw_data.head())
