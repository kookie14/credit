import logging
from joblib import dump, load
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from data.base_data import CreditDataModule
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score

class BaseGrid:
    def __init__(self, 
                 grid_search,
                 best_model=None
                 ):
        self.grid_search = grid_search
        self.best_model = best_model

    def grid(self, data: CreditDataModule):
        train_set = data.get_train_set()
        X_train = train_set.drop(columns=[data.class_column_name]).values
        y_train = train_set[data.class_column_name].values
        logging.info(f"Param grids: {(self.grid_search.param_grid)}")
        self.grid_search.param_grid = dict(self.grid_search.param_grid)
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        logging.info("Model fitted.")
        logging.info(f"Best parameters found: {self.grid_search.best_params_}")
        logging.info(f"Best score found: {self.grid_search.best_score_}")

    def validate(self, data: CreditDataModule):
        test_set = data.get_test_set()
        X_test = test_set.drop(columns=[data.class_column_name]).values
        y_test = test_set[data.class_column_name].values
        y_pred = self.best_model.predict(X_test).reshape(-1,1)
        y_pred_prob = self.best_model.predict_proba(X_test)[:, 1]
        score = accuracy_score(y_test.reshape(-1,1),y_pred)
        precision = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        recall = recall_score(y_test.reshape(-1,1), y_pred)
        logging.info(f"Model Accuracy = {score:.4f}\tPrecision = {precision:.4f}\tAUC-ROC = {auc:.4f}\tRecall = {recall:.4f}")

    def save_model(self, model_output_path):
        dump(self.net, model_output_path)

    def load_model(self, model_path):
        self.net = load(model_path)