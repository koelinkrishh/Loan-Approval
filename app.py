from flask import Flask, request, render_template, url_for

import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from src.Pipeline.Pipeline import CustomMapper, PredictPipeline

from src.Exception import CustomException
from src.Logging import logging

## Create base for app
application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
   return render_template("index.html")

@app.route('/prediction_data', methods=['GET','POST'])
def prediction_datapoint():
   if request.method == 'GET':
      return render_template("home.html")
   elif request.method == 'POST':
      try:
         # Extract data from form
         input_data = {key: request.form.get(key) for key in [
            'LoanID', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education',
            'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
            'LoanPurpose', 'HasCoSigner', 'Default']}
         
         # Convert to proper data types
         numerical_fields = {'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'}
         Integer_fields = {'HasMortgage', 'HasDependents', 'HasCoSigner'}
         
         
         for key, val in input_data.items():
            if val == '':
               input_data[key] = np.nan  # Replace missing values with NaN
            elif key in numerical_fields:
               input_data[key] = float(val)
            elif key in Integer_fields:
               input_data[key] = int(val) if val.isdigit() else 0
            
         # Now, create a CustomMapper instance
         data = CustomMapper(input_data)
         df = data.get_data_as_dataframe()
         
         ## Now make prediction onto input_data
         Pipe = PredictPipeline()
         ans = Pipe.predict(df)
         logging.info(ans)
         
         return render_template("home.html", results=ans)
      except Exception as e:
         raise CustomException(e, sys)


if __name__ == "__main__":
   app.run(host="127.0.0.1", debug=True)



