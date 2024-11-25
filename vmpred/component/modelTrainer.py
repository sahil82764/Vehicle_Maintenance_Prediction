from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import ModelTrainerConfig
from vmpred.entity.artifactEntity import DataTransformationArtifact
from sklearn.model_selection import GridSearchCV, train_test_split
import importlib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os, sys
import pickle


class ModelTrainer:
    
    def __init__(self, modelTrainerConfig=ModelTrainerConfig, dataTransformationArtifact=DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*30}Model Training log started{'<<'*30}\n\n")
            
            self.model_trainer_config = modelTrainerConfig
            self.data_transformation_artifact = dataTransformationArtifact

        except Exception as e:
            raise vmException(e,sys) from e
        
    def initiate_model_trainer(self):
        pass




    def __del__(self):
        logging.info(f"{'>>'*20}Data Transforamtion Log Completed{'<<'*20}\n\n")