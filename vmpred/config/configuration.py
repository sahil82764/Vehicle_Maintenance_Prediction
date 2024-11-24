from vmpred.entity.configEntity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from vmpred.util.util import read_yaml_file
from vmpred.logger import logging
import sys
import os
from vmpred.constant import *
from vmpred.exception import vmException

class Configuration:

    def __init__(self, config_file_path:str = CONFIG_FILE_PATH, current_time_stamp:str = CURRENT_TIME_STAMP ) -> None:
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            # self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise vmException(e,sys) from e
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:

            data_ingestion_info = self.config_info["data_ingestion_config"]

            raw_data_dir = os.path.join(ROOT_DIR,data_ingestion_info["raw_data_dir"])

            ingested_dir = os.path.join(ROOT_DIR,data_ingestion_info["data_dir"], data_ingestion_info["ingested_dir"])

            data_ingestion_config = DataIngestionConfig(
                raw_data_dir = raw_data_dir,
                ingested_dir = ingested_dir
            )

            logging.info(f"Data Ingestion Config: {data_ingestion_config} \n")
            return data_ingestion_config
        
        except Exception as e:
            raise vmException(e,sys) from e
        
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_info = self.config_info["data_validation_config"]

            schema_file_path = os.path.join( ROOT_DIR, data_validation_info["schema_dir"], data_validation_info["schema_file_name"])

            validated_dir = os.path.join(ROOT_DIR,data_validation_info["data_dir"], data_validation_info["validated_dir"])

            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                validated_dir = validated_dir
            )

            return data_validation_config
        
        except Exception as e:
            raise vmException(e,sys) from e
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            
            data_transformation_info = self.config_info["data_transformation_config"]

            transformed_dir = os.path.join( ROOT_DIR, data_transformation_info["data_dir"], data_transformation_info["transformed_dir"])
            os.makedirs(transformed_dir, exist_ok=True)

            train_dir = os.path.join( transformed_dir, data_transformation_info["transformed_train_dir"])
            test_dir = os.path.join( transformed_dir, data_transformation_info["transformed_test_dir"])

            data_transformation_config = DataTransformationConfig(
                transformed_train_dir= train_dir,
                transformed_test_dir= test_dir, 
                test_size=data_transformation_info["test_size"],
                random_state=data_transformation_info["random_state"]
            )

            return data_transformation_config


        except Exception as e:
            raise vmException(e,sys) from e
        
        