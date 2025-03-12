import os
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.Exception import CustomException
from src.Logging import logging

def hyperparameter_tuning(model_name, model, X_train, y_train):
   try:
      param_grid = {
         'Logistic Regression': {
            'max_iter':[100,300,500],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs', 'sag']
         },
         'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9],
         },
         'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30]
         },
         'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.5]
         },
         'Gradient Boost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
         },
         'XGBoost': {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [3, 5, 7],
            'gamma': [0, 0.2, 0.4],
         }
      }
      
      if model_name in param_grid:
         logging.info(f"Performing hyperparameter tuning for {model_name}")
         grid_search = GridSearchCV(model, param_grid[model_name], cv=3, scoring='accuracy', n_jobs=-1)
         grid_search.fit(X_train, y_train)
         logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
         print(f"Best params for {model_name}: {grid_search.best_params_}")
         return grid_search.best_estimator_
      else:
         logging.info(f"No hyperparameter tuning available for {model_name}")
         return model
   
   except Exception as e:
      raise CustomException(e, sys)
