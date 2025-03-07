# For creating logs
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
# Get path for logs
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
# Append log path to existing directory
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
   filename=LOG_FILE_PATH,
   level = logging.INFO,
   format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)


