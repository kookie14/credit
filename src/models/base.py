import logging
from joblib import dump, load
from sklearn.discriminant_analysis import StandardScaler
from data.base_data import CreditDataModule
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score

class BaseModel:
    def __init__(
        self,
        net
    ):
        logging.info(f"model = {net.__class__.__name__}")
        self.net = net
        self.scaler = StandardScaler()

    def train(self, data: CreditDataModule):
        train_set = data.get_train_set()
        X_train = train_set.drop(columns=[data.class_column_name]).values
        y_train = train_set[data.class_column_name].values
        # X_train = self.scaler.fit_transform(X_train)
        self.net.fit(X_train, y_train)
        logging.info("Model fitted.")

    def validate(self, data: CreditDataModule):
        test_set = data.get_test_set()
        X_test = test_set.drop(columns=[data.class_column_name]).values
        y_test = test_set[data.class_column_name].values
        score = self.net.score(X_test, y_test)
        y_pred = self.net.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        logging.info(f"Model score = {score:.4f}\tAccuracy = {acc:.4f}\tPrecision = {precision:.4f}\tAUC-ROC = {auc:.4f}\tRecall = {recall:.4f}")

    def save_model(self, model_output_path):
        dump(self.net, model_output_path)

    def load_model(self, model_path):
        self.net = load(model_path)