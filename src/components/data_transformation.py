import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation based on the different types of data
        '''
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), #handling missing values
                    ("scaler", StandardScaler()) #standardization
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), #handling missing values
                    ("encoder", OneHotEncoder()), #encoding categorical variables
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")

            logging.info(f"Categorical columns : {categorical_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Train and test data loaded")

            logging.info("Obtaining preprocessing object...")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["reading_score", "writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe...")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saving preprocessor object to disk...")

            save_objects(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)

            
