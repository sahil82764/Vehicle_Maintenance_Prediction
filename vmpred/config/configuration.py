from vmpred.entity.configEntity import DataIngestionConfig
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

            logging.info(f"Data Ingestion Config: {data_ingestion_config}")
            return data_ingestion_config
        
        except Exception as e:
            raise vmException(e,sys) from e