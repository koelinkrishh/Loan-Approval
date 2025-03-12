import os
import sys
from src.Exception import CustomException
from src.Logging import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.Components.feature_engineering import DataTransformation
from src.Components.feature_engineering import DataTransformationConfig

from src.Components.model_training import ModelTrainerConfig
from src.Components.model_training import ModelTrainer

## Class to track down input data
@dataclass
class DataIngestionConfig:
   train_data_path: str=os.path.join('artifacts','train.csv')
   test_data_path: str=os.path.join('artifacts','test.csv')
   raw_data_path: str=os.path.join('artifacts', 'data.csv')

""" 
Dataclass is a decorator which allows you to create custom 
classes in Python. It is used to create a class with 
attributes that are defined in the class definition.
-> No need to init
"""

class DataIngestion:
   def __init__(self):
      self.ingestion_config = DataIngestionConfig()
   #-> All paths will automatically be saved inside ingestion_config variable
   
   def initiate_data_ingestion(self):
      logging.info("Entered data ingestion method or component")
      try:
         BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Move up 3 levels to the project root
         dataset_path = os.path.join(BASE_DIR, "Dataset", "Loan_default.csv")
         print(dataset_path)
         df = pd.read_csv(dataset_path)
         logging.info("Read the dataset as dataframe")
         
         ## Create the directory for the training data if it doesn't exist
         os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
         # Send raw data into its location
         df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
         
         ## Splitting Dataset
         logging.info("Train Test split initiated")
         train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
         train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
         test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
         logging.info("Ingestion of the data is completed")
         
         return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
         
      except Exception as e:
         raise CustomException(e, sys)


if __name__=="__main__":
   obj = DataIngestion()
   train_data, test_data = obj.initiate_data_ingestion()
   # print(train_data, test_data)
   
   data_trans = DataTransformation()
   train_arr,test_arr,_ = data_trans.Initiate_data_transformation(train_data, test_data)
   
   Model_trainer = ModelTrainer()
   result = Model_trainer.Initiate_model_trainer(train_arr, test_arr)
   print("Accuracy Score for Best model: ",result)
   
