import os
import sys
import shutil
from vmpred.entity.configEntity import DataIngestionConfig
from vmpred.entity.artifactEntity import DataIngestionArtifact
from vmpred.exception import vmException
from vmpred.logger import logging



class DataIngestion:
    
    def __init__(self, dataIngestionConfig: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion Log Started.{'<<'*20}")
            self.data_ingestion_config = dataIngestionConfig
        except Exception as e:
            raise vmException(e,sys) from e
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            rawDataDir = self.data_ingestion_config.raw_data_dir

            ingestedDataDir = self.data_ingestion_config.ingested_dir

            os.makedirs(ingestedDataDir, exist_ok=True)

            csv_files = [f for f in os.listdir(rawDataDir) if f.endswith('.csv')]
            if not csv_files:
                print("No CSV files found in the source directory.")
                return
            
            # Copy each file to the destination directory
            for file in csv_files:
                source_path = os.path.join(rawDataDir, file)
                destination_path = os.path.join(ingestedDataDir, file)
                shutil.copy(source_path, destination_path)


            logging.info(f"Ingesting file from: [{rawDataDir}] into: [{ingestedDataDir}] is now completed")

            data_ingestion_artifact = DataIngestionArtifact(
                ingested_data_dir = ingestedDataDir,
                is_ingested = True,
                message = f"Data ingestion completed successfully"
                )

            logging.info(f"Data ingestion artifact: [{data_ingestion_artifact}]")
            return data_ingestion_artifact


        except Exception as e:
            raise vmException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion Log Completed{'<<'*20}\n\n")

