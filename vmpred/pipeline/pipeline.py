from vmpred.entity.artifactEntity import DataIngestionArtifact, DataValidationArtifact
from vmpred.entity.configEntity import DataIngestionConfig
from vmpred.component.dataIngestion import DataIngestion, DataValidation

import os
import sys
from vmpred.config.configuration import Configuration
from vmpred.logger import logging
from vmpred.exception import vmException
from threading import Thread

class Pipeline(Thread):
    def __init__(self, config: Configuration) -> None:
        try:
            # os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            self.config = config
        except Exception as e:
            raise vmException(e,sys) from e
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(dataIngestionConfig = self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise vmException(e,sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact ) -> DataValidationArtifact:
        try:

            data_validation = DataValidation(
                data_validation_config=self.config.get_data_validation_config(),
                data_ingestion_artifact=data_ingestion_artifact
            )

            return data_validation.initiate_data_validation()

        except Exception as e:
            raise vmException(e,sys) from e
        
    def run_pipeline(self):
        try:

            logging.info("Pipeline Starting.")

            data_ingestion = self.start_data_ingestion()

            logging.info("Pipeline Completed.")

        except Exception as e:
            raise vmException(e,sys) from e

    
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise vmException(e,sys) from e
        
