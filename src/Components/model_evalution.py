from src.Exception import CustomException
import os
import sys

from sklearn.metrics import accuracy_score

def evaluate_models(X_train,X_test, y_train,y_test, models)-> dict:
   try:
      report = {}
      
      Algos_names = list(models.keys())
      Algos = list(models.values())
      
      for i,m in enumerate(Algos):
         model = m
         model.fit(X_train, y_train)
         y_train_pred = model.predict(X_train)
         y_test_pred = model.predict(X_test)
         train_model_score = accuracy_score(y_train, y_train_pred)
         test_model_score = accuracy_score(y_test, y_test_pred)
         report[Algos_names[i]] = test_model_score
      
      return report
   
   except Exception as e:
      raise CustomException(e, sys)
