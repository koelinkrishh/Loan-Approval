import os
import sys

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.Exception import CustomException
from src.Logging import logging

from src.Components.feature_engineering import DataTransformation
from src.Components.feature_engineering import DataTransformationConfig

from src.Components.model_evalution import evaluate_models
from src.Components.hyperparameters import hyperparameter_tuning

from src.Components.utils import save_object

@dataclass
class ModelTrainerConfig:
   trained_model_file_path=os.path.join('artifacts','model.pkl')

## Class to outline object who will do all the model training
class ModelTrainer:
   def __init__(self):
      self.model_trainer_config = ModelTrainerConfig()
   
   def Initiate_model_trainer(self, train_array, test_array):
      try:
         logging.info("Splitting Test and Train array.")
         X_train,X_test, y_train,y_test = (train_array[:,:-1], test_array[:,:-1], 
            train_array[:,-1], test_array[:,-1])
         
         
         models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            # 'SVM': SVC(),
            'Guassian': GaussianNB(),
            # 'KNN': KNeighborsClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            # 'Gradient Boost': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier()
         }
         
         model_report:dict = evaluate_models(X_train,X_test, y_train,y_test, models=models)
         
         ## Get best model and use it to make predictions
         best_model_score = max(sorted(model_report.values()))
         best_model_index = list(model_report.values()).index(best_model_score)
         best_model_name = list(model_report.keys())[best_model_index]
         best_model = models[best_model_name]
         
         
         if best_model_score<0.6:
            raise CustomException("No Best model found")
         logging.info("Best model found on both training and testing dataset")
         
         ## Hyperparameter tuning for the best model
         best_model = hyperparameter_tuning(best_model_name ,best_model, X_train, y_train)
         
         save_object(
            file_path = self.model_trainer_config.trained_model_file_path,
            obj = best_model
         )
         
         predicted = best_model.predict(X_test)
         acc = accuracy_score(y_test, predicted)
         print("Best Model is ", best_model_name," with accuracy score: ",acc)
         
         return acc
      except Exception as e:
         raise CustomException(e, sys)
   











