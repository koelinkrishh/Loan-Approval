# For exception handling
import sys
import os

def error_message_detail(error, error_detail:sys):
   _,_,exc_tb = error_detail.exc_info()
   file_name = exc_tb.tb_frame.f_code.co_filename
   error_message = f"Error occured in python script name [{file_name}],line number [{exc_tb.tb_lineno}]: error message [{str(error)}]"
   return error_message

class CustomException(Exception):
   def __init__(self, error_message, error_detail:sys):
      # Inherited from exception class
      super().__init__(error_message) # get base error class message
      self.error_msg = error_message_detail(error_message, error_detail=error_detail)
      
   
   def __str__(self): # Prints error message
      """Return the error message string representation of the exception."""
      return self.error_msg
      

