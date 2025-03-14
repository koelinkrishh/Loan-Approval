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
            'max_iter': [100, 300, 500],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'l1_ratio': [0.1, 0.5, 0.9],  # Used when 'elasticnet' is chosen
         },
         'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],  # Added distance metrics
            'p': [1, 2]  # p=1 (Manhattan), p=2 (Euclidean) for Minkowski distance
         },
         'Decision Tree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 20, 30, 50],  
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4],  # Added minimum samples per leaf
            'max_features': ['sqrt', 'log2', None],
         },
         'AdaBoost': {
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R'],
         },
         'Gradient Boost': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],  
            'subsample': [0.5, 0.7, 1.0],
            'min_samples_split': [2, 5, 10],  # Minimum samples per split
            'min_samples_leaf': [2, 4],  # Minimum samples per leaf
         },
         'XGBoost': {
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
            'max_depth': [3, 5, 7, 10,], 
            'min_child_weight': [1, 3, 5, 7], 
            'gamma': [0, 0.2, 0.4], 
            'subsample': [0.6, 0.8, 1.0],  # Fraction of samples per boosting round
            'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features for each tree
            # 'lambda': [0, 1, 5],  # L2 regularization (Ridge)
            # 'alpha': [0, 1, 5],  # L1 regularization (Lasso)
         }
      }
      
      if model_name in param_grid:
         logging.info(f"Performing hyperparameter tuning for {model_name}")
         grid_search = RandomizedSearchCV(model, param_grid[model_name], n_iter=50,cv=3, scoring='accuracy', n_jobs=-1)
         grid_search.fit(X_train, y_train)
         logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
         print(f"Best params for {model_name}: {grid_search.best_params_}")
         return grid_search.best_estimator_
      else:
         logging.info(f"No hyperparameter tuning available for {model_name}")
         return model
   
   except Exception as e:
      raise CustomException(e, sys)
