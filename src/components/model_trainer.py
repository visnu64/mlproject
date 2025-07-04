import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customexception
from src.logger import logging
from src.utilis import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor()
            } 

            params = {
                "decisiontree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

                },
                "randomforest": {
                    'n_estimators': [50,8,16,32,75,82,100, 200],
                    
                },
                "gradientboosting": {
                    'learning_rate': [0.01,0.05, 0.1, 0.2, 0.3],
                    'n_estimators': [50,8,16,32,75,82,100, 200],
                    'subsample': [0.8, 0.9, 1.0]
                },
                "linearregression": {},
                "knn": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                "xgboost": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'n_estimators': [50,8,16,32,75,82,100, 200],
                    'subsample': [0.8, 0.9, 1.0]
                },
                "adaboost": {
                    'n_estimators': [50,8,16,32,75,82,100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
                }


            }

            model_report = evaluate_models(
                x_train=x_train, 
                y_train=y_train,
                x_test=x_test, 
                y_test=y_test,
                models=models,
                params=params
            )
            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise customexception("no best models found")
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(x_test)

            r2_square=r2_score(y_test, predicted)
            return r2_square
        

        except Exception as e:
            raise customexception(e,sys)