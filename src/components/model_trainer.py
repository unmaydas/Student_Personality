import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Models to train
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(eval_metric="logloss"),
                "AdaBoost": AdaBoostClassifier(),
            }

            # Hyperparameters for tuning
            params = {

                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "n_estimators": [50, 100, 200],
                    "subsample": [0.8, 0.9, 1.0],
                    "max_depth": [3, 5]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 1],
                    "n_estimators": [50, 100, 200]
                }

            }

            logging.info("Starting model training and hyperparameter tuning")

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            logging.info(f"Model Report: {model_report}")

            # Select best model based on test score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy", sys)

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")
            # Train best model
            best_model.fit(X_train, y_train)

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            # Predictions
            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average="weighted")
            recall = recall_score(y_test, predicted, average="weighted")
            f1 = f1_score(y_test, predicted, average="weighted")

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")

            return accuracy, precision, recall, f1

        except Exception as e:
            raise CustomException(e, sys)