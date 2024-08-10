import os
import sys
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTrasformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTrasformation: 
    def __init__(self):
        self.data_trasformation_config = DataTrasformationConfig()

    def get_data_trasformer_object(self):
        '''
        This is the function that would be responsible for data transformation
        '''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()), 
                    ("scaler", StandardScaler(with_mean = False))
                ]
            )
            
            logging.info("Numerical features Standard Scaling completed successfully")
            logging.info("Categorical features Encoding completed successfully")

            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )
            
            logging.info("Data Preprocessing pipeline created successfully")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train & test data completed successfully")
            logging.info("Obtaining preprocessing objects")

            preprocessing_obj = self.get_data_trasformer_object()

            target_columns_names = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_columns_names], axis = 1)
            target_feature_train_df = train_df[target_columns_names]

            input_feature_test_df = test_df.drop(columns = [target_columns_names], axis = 1)
            target_feature_test_df = test_df[target_columns_names]

            logging.info(f"Applying preprocessing on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object (

                file_path = self.data_trasformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj 

            )

            return (

                train_arr, 
                test_arr, 
                self.data_trasformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e, sys)