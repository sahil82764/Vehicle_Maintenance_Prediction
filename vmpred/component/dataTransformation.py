from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import DataTransformationConfig
from vmpred.entity.artifactEntity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
import sys, os
import numpy as np
import pandas as pd
from vmpred.constant import *
from vmpred.util.util import read_parquet

# vehicle_model: category
# mileage: integer
# maintenance_history: category
# reported_issues: integer
# vehicle_age: integer
# fuel_type: category
# transmission_type: category
# engine_size: integer
# odometer_reading: integer
# last_service_date: date
# warranty_expiry_date: date
# owner_type: category
# insurance_premium: float
# service_history: integer
# accident_history: integer
# fuel_efficiency: float
# tire_condition: category
# brake_condition: category
# battery_status: category
# need_maintenance: integer


class DataTransformation:
    
    def __init__(self, dataTransformationConfig: DataTransformationConfig, dataIngestionArtifact:DataIngestionArtifact, dataValidationArtifact:DataValidationArtifact):
        try:
            
            logging.info(f"{'>>'*30}Data Transforamtion log started{'<<'*30}\n\n")
            self.data_transformation_config = dataTransformationConfig
            self.data_ingestion_artifact = dataIngestionArtifact
            self.data_validation_artifact = dataValidationArtifact

        except Exception as e:
            raise vmException(e,sys) from e
        
    def handle_missing_values(self, df: pd.DataFrame):
        try:
            logging.info(f"{df.isnull().sum()}")
            return df.dropna()
        except Exception as e:
            raise vmException(e,sys) from e
        
    def drop_duplicates(self, df: pd.DataFrame):
        try:
            logging.info("Duplicates Dropped if any")
            return df.drop_duplicates()
        except Exception as e:
            raise vmException(e,sys) from e
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        
        try:
            validated_file_path = self.data_validation_artifact.validated_file_path
            vmData = read_parquet(validated_file_path)

            vmData = self.handle_missing_values(vmData)
            vmData = self.drop_duplicates(vmData)
            vmData = self.feature_enginnering(vmData)

        except Exception as e:
            raise vmException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*20}Data Transforamtion Log Completed{'<<'*20}\n\n")