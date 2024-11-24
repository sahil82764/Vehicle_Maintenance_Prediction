from vmpred.entity.artifactEntity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from vmpred.entity.configEntity import DataIngestionConfig
from vmpred.component.dataIngestion import DataIngestion 
from vmpred.component.dataValidation import DataValidation
from vmpred.component.dataTransformation import DataTransformation
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
        
    def start_data_validation(self, DataIngestionArtifact: DataIngestionArtifact ) -> DataValidationArtifact:
        try:

            data_validation = DataValidation(
                dataValidationConfig=self.config.get_data_validation_config(),
                dataIngestionArtifact=DataIngestionArtifact
            )

            return data_validation.initiate_data_validation()

        except Exception as e:
            raise vmException(e,sys) from e
        
    def start_data_transformation(self, DataIngestionArtifact: DataIngestionArtifact, DataValidationArtifact = DataValidationArtifact) -> DataTransformationArtifact:
        
        try:

            data_transformation = DataTransformation(
                dataTransformationConfig=self.config.get_data_transformation_config(),
                dataIngestionArtifact=DataIngestionArtifact,
                dataValidationArtifact=DataValidationArtifact
            )

            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise vmException(e,sys) from e
        
    def run_pipeline(self):
        try:

            logging.info("Pipeline Starting.")

            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(DataIngestionArtifact = data_ingestion_artifact)

            logging.info("Pipeline Completed.")

        except Exception as e:
            raise vmException(e,sys) from e

    
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise vmException(e,sys) from e
        
