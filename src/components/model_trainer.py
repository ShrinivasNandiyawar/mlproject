import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects,evaulate_models


@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts', "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, preprocessor_obj_path):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "linear_regression" : LinearRegression(),
                "decision_tree" : DecisionTreeRegressor(),
                "random_forest" : RandomForestRegressor(),
                "adaboost" : AdaBoostRegressor(),
                "gradient_boosting" : GradientBoostingRegressor(),
                "xgboost" : XGBRegressor(),
                "catboost" : CatBoostRegressor(logging_level='Silent'),
                "knn" : KNeighborsRegressor()
            }

            model_report:dict = evaulate_models(X_train = x_train, Y_train = y_train, 
                                               X_test = x_test, Y_test = y_test,  
                                               models = models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Training best model: {best_model_name} and the model score : {best_model_score}")

            save_objects(
                file_path=self.model_trainer_config.trainer_model_file_path, 
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted) 

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
