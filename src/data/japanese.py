import logging
import pandas as pd
from base_data import Dataset
import numpy as np

class JapaneseDataset(Dataset):
    def __init__(self, data_path: str, class_column_name: str = 'class'):
        super().__init__(data_path, class_column_name)
        logging.info(f"data_path = {self.data_path}")
        self.raw_data = pd.read_csv(data_path, header=None, sep=",")
        logging.info(f"raw_data shape = {self.raw_data.shape}")
        self._set_column_names()
        self._remove_missing_values()

    
    def _set_column_names(self):
        self.raw_data.columns = ["A"+ str(i) for i in range(1, 16)] + [self.class_column_name]

    def _remove_missing_values(self):
        #replace ? with NaN
        self.raw_data = self.raw_data.replace("?", np.nan)
        logging.info(f"has ? = {(self.raw_data == '?').values.any()}")
        logging.info(f"has NaNs = {self.raw_data.isnull().values.any()}") 
        # drop rows with NaN values
        self.raw_data = self.raw_data.dropna()
    
    def _get_dummies(self):
        self.raw_data[13] = self.raw_data[13].astype('float32')
        self.raw_data[10] = self.raw_data[10].astype('float32')
        self.raw_data = pd.get_dummies(self.raw_data, dtype='float3',
                                        columns=[0, 3, 4, 5, 6, 8, 9, 11, 12, 15])
        
        self.raw_data['15_+'] =self.raw_data['15_+'].astype('int32')
        self.raw_data = self.raw_data.drop(columns='15_-')
        


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    japanese_data = JapaneseDataset(
        data_path='/home/cuctt/credit/data/Japanese/crx.data', 
        class_column_name='class'
    )
    print(japanese_data.raw_data)
