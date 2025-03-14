## In this file, we will cearte a pipeline for entire model
import sys
import os

import numpy as np
import pandas as pd

from src.Exception import CustomException
from src.Components.utils import load_object

class PredictPipeline:
   def __init__(self):
      pass
   
   def predict(self, features):
      try:
         model_path = os.path.join('artifacts','model.pkl')
         preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
         
         # Ensure files exist before loading
         if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
         if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
         
         
         ## Load object just loads our pickle file with model
         model = load_object(file_path = model_path)
         preprocessor = load_object(file_path = preprocessor_path)
         
         ## Perform feature engineering on input data
         Input = preprocessor.transform(features)
         ## Making predict on Input data from app using model pipeline
         pred = model.predict(Input)
         
         return pred
      except Exception as e:
         raise CustomException(e, sys)

"""
Now, we will create a custom class responsible for mapping all the inputs
that we are giving in the HTML to backend with particular values
"""
class CustomMapper:
   def __init__(self, variable: dict):
      ## variables is an dict to store all input variables
      self.input = variable
      
      ## Unpack variables
      """
      self.LoanId = self.input['LoanId']
      self.Age = self.input['Age']
      self.Income = self.input['Income']
      self.LoanAmount = self.input['LoanAmount']
      self.CreditScore = self.input['CreditScore']
      self.MonthsEmployed = self.input['MonthsEmployed']
      self.NumCreditLines = self.input['NumCreditLines']
      self.InterestRate = self.input['InterestRate']
      self.LoanTerm = self.input['LoanTerm']
      self.DTIRatio = self.input['DTIRatio']
      self.Education = self.input['Education']
      self.EmploymentType = self.input['EmploymentType']
      self.MaritalStatus = self.input['MaritalStatus']
      self.HasMortgage = self.input['HasMortgage']
      self.HasDependents = self.input['HasDependents']
      self.LoanPurpose = self.input['LoanPurpose']
      self.HasCoSigner = self.input['HasCoSigner']
      self.Default = self.input['Default']
      """
      ## OR Unpack variables dynamically
      for key, value in self.input.items():
         setattr(self, key, value) # setter
   
   def get_data_as_dataframe(self):
      try:
         if "Default" in self.input:
            del self.input['Default']
         return pd.DataFrame([self.input])
      
      except Exception as e:
         raise CustomException(e, sys)

