import os
import sys
from src.exception import customexception
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import datatransformation
from src.components.data_transformation import datatransformationconfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class dataingestconfig:
    train_data_path: str= os.path.join('artifacts',"train.csv")
    test_data_path: str= os.path.join('artifacts',"test.csv")
    raw_data_path: str= os.path.join('artifacts',"data.csv")

class dataingestion:
    def __init__(self):
        self.ingestion_config=dataingestconfig()

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            data_path = os.path.join("src", "notebook", "data", "StudentsPerformance.csv")
            df = pd.read_csv(data_path)
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise customexception(e,sys)
        
if __name__=="__main__":
    obj = dataingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = datatransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
    # You can now use train_arr, test_arr, preprocessor_path as needed
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
