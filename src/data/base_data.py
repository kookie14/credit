import logging 
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, data_path : str, class_column_name: str = 'class'):
        self.data_path = data_path
        self.class_column_name = class_column_name
        self.raw_data: pd.DataFrame = None

class CreditDataModule:
    def __init__(
        self,
        dataset: Dataset,
        class_column_name: str = "class",
        seed: int = 42,
        test_size: float = 0.25
    ): 
        self.datatset = dataset
        self.class_column_name = class_column_name
        self.seed = seed
        self.test_size = test_size
        self.train_set, self.test_set = self._split_dataset()

    def get_train_set(self) -> pd.DataFrame:
        return self.train_set
    
    def get_test_set(self) -> pd.DataFrame:
        return self.test_set

    def _split_dataset(self):
        logging.info(f'Splitting dataset... test_size = {self.test_set} seed = {self.seed}')
        train_set, test_set = train_test_split(self.datatset.raw_data, test_size=self.test_size, random_state=self.seed)
        return train_set, test_set
        

