import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from src.Exception import CustomException
from src.Logging import logging
from src.Components.utils import save_object, load_object

@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
   def __init__(self):
      self.data_transformation_config = DataTransformationConfig()
      self.binning_model = None  # Store binning model globally
      self.scaler = MinMaxScaler()  # Store MinMaxScaler globally
   
   def bin_and_scale_age(self, dataset):
      if not isinstance(dataset, pd.DataFrame):
         dataset = pd.DataFrame(dataset, columns=['Age'])
      
      dataset = dataset.values.reshape(-1, 1)  # Ensure it's 2D
      
      if self.binning_model is None:
         self.binning_model = KBinsDiscretizer(n_bins=14, encode='ordinal', strategy='uniform')
         dataset = self.binning_model.fit_transform(dataset)
      else:
         dataset = self.binning_model.transform(dataset)
      
      if not hasattr(self, "scaler_fitted"):
         self.scaler.fit(dataset)
         self.scaler_fitted = True
      
      return self.scaler.transform(dataset)
   
   def get_data_transformer_object(self):
      try:
         binary_features = ['HasCoSigner', 'HasDependents', 'HasMortgage']
         ordinal_features = ['Education']
         nominal_features = ['MaritalStatus', 'LoanPurpose']
         label_encoded_features = ['EmploymentType']
         numerical_features = ['Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm']
         
         def BinaryEncoder(dataset):
            if not isinstance(dataset, pd.DataFrame):
               dataset = pd.DataFrame(dataset, columns=binary_features)
            for col in binary_features:
               dataset[col] = dataset[col].astype(str).map({'Yes': 1, 'No': 0})
            return dataset
         
         def Label_encode(dataset):
            if not isinstance(dataset, pd.DataFrame):
               dataset = pd.DataFrame(dataset, columns=label_encoded_features)
            for col in label_encoded_features:
               if col not in self.label_encoders:
                  self.label_encoders[col] = LabelEncoder()
                  dataset[col] = self.label_encoders[col].fit_transform(dataset[col])
               else:
                  dataset[col] = self.label_encoders[col].transform(dataset[col])
            return dataset
         
         self.label_encoders = {}
         
         FT_binary = Pipeline(steps=[
            ('binary_imputer', SimpleImputer(strategy='most_frequent')),
            ('binary_transformer', FunctionTransformer(BinaryEncoder, validate=False))
         ])
         FT_label = Pipeline(steps=[
            ('label_imputer', SimpleImputer(strategy='most_frequent')),
            ('label_transformer', FunctionTransformer(Label_encode, validate=False))
         ])
         FT_age = Pipeline(steps=[
            ('age_imputer', SimpleImputer(strategy='median')),
            ('age_transformer', FunctionTransformer(self.bin_and_scale_age, validate=False))
         ])
         
         numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
         ])
         OHE_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
         ])
         Ordinal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=[["High School", "Bachelor's", "Master's", "PhD"]]))
         ])
         
         Ct = ColumnTransformer(transformers=[
            ('num_pipeline', numerical_pipeline, numerical_features),
            ('binary_pipeline', FT_binary, binary_features),
            ('label_pipeline', FT_label, label_encoded_features),
            ('OHE_pipeline', OHE_pipeline, nominal_features),
            ('Ordinal_pipeline', Ordinal_pipeline, ordinal_features),
            ('age_pipeline', FT_age, ['Age'])
         ], remainder='passthrough', n_jobs=-1)
         
         return Pipeline(steps=[('column_transformer', Ct)])
      
      except Exception as e:
         raise CustomException(e, sys)
   
   
   def Initiate_data_transformation(self, train_path, test_path):
      try:
         dataset = pd.read_csv(train_path)
         test_df = pd.read_csv(test_path)
         logging.info("Reading train and test data completed")
         
         if "LoanID" in dataset.columns:
            dataset = dataset.drop(columns=["LoanID"], axis=1)
         if "LoanID" in test_df.columns:
            test_df = test_df.drop(columns=["LoanID"], axis=1)
         
         logging.info("Obtaining preprocessing object")
         self.preprocessing_obj = self.get_data_transformer_object()
         
         Target = "Default"
         input_feature_train = dataset.drop(columns=[Target], axis=1)
         input_feature_test = test_df.drop(columns=[Target], axis=1)
         target_feature_train = dataset[[Target]]
         target_feature_test = test_df[[Target]]
         
         logging.info("Applying preprocessing object on training and testing datasets.")
         Input_train_arr = self.preprocessing_obj.fit_transform(input_feature_train)
         Input_test_arr = self.preprocessing_obj.transform(input_feature_test)
         
         train_arr = np.c_[Input_train_arr, np.array(target_feature_train)]
         test_arr = np.c_[Input_test_arr, np.array(target_feature_test)]
         
         logging.info("Saved preprocessing object.")
         save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=self.preprocessing_obj
         )
         print("Preprocessing object saved successfully.")
         return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
      
      except Exception as e:
         print(f"Error saving object: {e}")
         raise CustomException(e, sys)