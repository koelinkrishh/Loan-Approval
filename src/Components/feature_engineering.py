import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


from src.Exception import CustomException
from src.Logging import logging

from src.Components.utils import save_object

@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
   def __init__(self):
      self.data_transformation_config = DataTransformationConfig()
   
   def get_transformed_column_names(ct, original_features):
      """Extract transformed column names from ColumnTransformer."""
      output_features = []
      
      for name, transformer, features in ct.transformers_:
         if transformer == 'passthrough':
            output_features.extend(features)
         elif transformer == 'drop':
            continue
         elif hasattr(transformer, "get_feature_names_out"):
            output_features.extend(transformer.get_feature_names_out(features))
         else:
            # Handle FunctionTransformer case
            output_features.extend([f"{name}_{i}" for i in range(len(features))])
            
      return output_features
   
   def get_data_transformer_object(self):
      """ 
      This function is responsible for data transformation
      """
      try:
         ## Defining Feature Classes
         binary_features = ['HasCoSigner', 'HasDependents','HasMortgage']
         ordinal_features = ['Education']
         nominal_features = ['MaritalStatus','LoanPurpose']
         label_encoded_features = ['EmploymentType']
         numerical_features = ['Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm']
         
         ## Custom function for binary encoding:
         def BinaryEncoder(dataset):
            for col in binary_features:
               dataset[col] = dataset[col].map({'Yes':1, 'No':0})
            return dataset
         
         def Label_encode(dataset):
            return dataset.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype=='O' else col)
         
         ## Custom transformer for Age as it needs both discretization and scaling
         def bin_and_scale_age(dataset):
            if dataset.ndim == 1: # Reshape into 2D
               dataset = dataset.reshape(-1, 1)
            
            # Step-1: Discretization
            kbins = KBinsDiscretizer(n_bins=14, encode='ordinal', strategy='uniform')
            dataset = kbins.fit_transform(dataset)
            
            # Step-2: Scale the binned values
            scaler = MinMaxScaler()
            dataset = scaler.fit_transform(dataset)
            
            return dataset
         
         ## Function Transformers
         FT_binary = FunctionTransformer(BinaryEncoder, validate=False)
         FT_label = FunctionTransformer(Label_encode, validate=False)
         FT_age = FunctionTransformer(bin_and_scale_age, validate=False)
         
         ## Defining Column Transformer
         Ct = ColumnTransformer(transformers=[
            ('binary', FT_binary, binary_features),
            ('ordinal', OrdinalEncoder(categories=[["High School","Bachelor's","Master's","PhD"]]), ordinal_features),
            ('Nominal->OHE', OneHotEncoder(drop='first'), nominal_features),
            ('Label->LE', FT_label, label_encoded_features),
            ('Age Transformer', FT_age, ['Age']),
            ('Numerical Scaling-> MinMax', MinMaxScaler(), numerical_features),
         ]  ,remainder='passthrough',
            force_int_remainder_cols = False, # This ensures column names remain correctly
         )
         
         ## Creating a Pipeline
         pipeline = Pipeline(steps=[
            #('imputer', SimpleImputer(strategy='most_frequent')),
            ('column_transformer', Ct)
         ])
         
         return pipeline
      except Exception as e:
         raise CustomException(e, sys)
   
   def Initiate_data_transformation(self, train_path, test_path):
      ## Train,test path come from data ingestion object
      try:
         training_df = pd.read_csv(train_path)
         test_df = pd.read_csv(test_path)
         
         
         logging.info("Reading train and test data completed")
         
         ## Remember to drop Loan ID column
         if "LoanID" in training_df.columns:
            training_df = training_df.drop(columns=["LoanID"], axis=1)
         if "LoanID" in test_df.columns:
            test_df = test_df.drop(columns=["LoanID"], axis=1)
         
         logging.info("Obtaining preprocessing object")
         
   ## Object which will perform all given functions -> preprocessor_obj gets pipeline
         self.preprocessing_obj = self.get_data_transformer_object()
         
         Target = "Default" # Target column name
         
         input_feature_train = training_df.drop(columns=[Target], axis=1)
         input_feature_test = test_df.drop(columns=[Target], axis=1)
         target_feature_train = training_df[Target]
         target_feature_test = test_df[Target]
         
         logging.info("Applying preprocessing object on training and testing datasets.")
         
         # # Ensure input is a DataFrame before transformation
         # input_feature_train = pd.DataFrame(input_feature_train)
         # input_feature_test = pd.DataFrame(input_feature_test)
         
         Input_train_arr = self.preprocessing_obj.fit_transform(input_feature_train)
         Input_test_arr = self.preprocessing_obj.fit_transform(input_feature_test)
         
         train_arr = np.c_[Input_train_arr, np.array(target_feature_train)]
         test_arr = np.c_[Input_test_arr, np.array(target_feature_test)]
         
         logging.info("Saved preprocessing object.")
         
         ## Save preprocessor object as a pickle file
         save_object(
            file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj=self.preprocessing_obj
         )
         
         return (train_arr,test_arr, self.data_transformation_config.preprocessor_obj_file_path)
      except Exception as e:
         raise CustomException(e, sys)



