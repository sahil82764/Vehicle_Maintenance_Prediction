from vmpred.logger import logging
from vmpred.exception import vmException
from vmpred.entity.configEntity import DataValidationConfig
from vmpred.entity.artifactEntity import DataIngestionArtifact, DataValidationArtifact
from vmpred.util.util import read_yaml_file
import os, sys
import glob
import pandas as pd

class DataValidation:

    def __init__(self, dataValidationConfig:DataValidationConfig, dataIngestionArtifact:DataIngestionArtifact):
        try:
            
            logging.info(f"{'>>'*30}Data Validation log started{'<<'*30}\n\n")
            self.data_validation_config = dataValidationConfig
            self.data_ingestion_artifact = dataIngestionArtifact

        except Exception as e:
            raise vmException(e,sys) from e
        
    def validate_and_compile_data(self, ):
        try:
            ingested_dir = self.data_ingestion_artifact.ingested_data_dir
            validated_dir = self.data_validation_config.validated_dir
            schema = read_yaml_file(file_path=self.data_validation_config.schema_file_path)
            
            os.makedirs(validated_dir, exist_ok=True) # Create validated directory if it doesn't exist

            columns = schema["columns"]
            column_names = list(columns.keys()) # Get column names from schema

            #Use glob to get all CSV files in ingested_dir
            csv_files = glob.glob(os.path.join(ingested_dir,"*.csv"))

            if not csv_files:
                raise Exception(f"No CSV files found in ingested directory: {ingested_dir}")

            # Compile all CSV files into a single DataFrame
            compiled_df = pd.DataFrame(columns=column_names)
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)

                    # Lowercase column names
                    df.columns = map(str.lower, df.columns)
                    
                    # Validate columns and data types. Neglect file on discrepancy
                    try:
                        self._validate_columns(df,column_names)
                        self._validate_dtypes(df,columns) 
                        compiled_df = pd.concat([compiled_df,df], ignore_index=True)
                        logging.info(f"Validation successful for file: {file_path}")

                    except Exception as e:
                        logging.warning(f"Validation failed for file {file_path}. Error: {e}. Neglecting this file.")
                        #Do not add to compiled_df

                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
                    #You might want to raise exception here depending on your failure tolerance


            #Save the compiled dataframe. Check if there is any data
            if compiled_df.empty:
                raise Exception("No valid data found after validation. Compilation failed.")

            parquet_file_path = os.path.join(validated_dir, "validated_data.parquet")
            compiled_df.to_parquet(parquet_file_path, engine="pyarrow")

            # csv_file_path = os.path.join(validated_dir, "validated_data.csv")
            # compiled_df.to_csv(csv_file_path, index=False)


            logging.info(f"Validated and compiled data saved to: {parquet_file_path}")
            return parquet_file_path
            # return csv_file_path

        except Exception as e:
            raise vmException(e,sys) from e
        
    def _validate_columns(self,df:pd.DataFrame,column_names:list):
        # Convert column names to lowercase for case-insensitive comparison
        df_cols_lower = set(df.columns)
        column_names_lower = set(column_names)


        if not column_names_lower.issubset(df_cols_lower):
            missing_columns = set(column_names_lower) - set(df_cols_lower)
            raise Exception(f"Following columns are missing in csv file (case-insensitive): {missing_columns}")

    
    def _validate_dtypes(self, df: pd.DataFrame, schema: dict):
        dtype_mapping = {
            "integer": "Int64",
            "float": "float64",
            "category": "category",
            "date": "datetime64[ns]"
        }

        for col, dtype in schema.items():
            expected_dtype = dtype_mapping.get(dtype)  # Use get() to handle missing keys
            if expected_dtype is None:
                logging.warning(f"Unknown dtype '{dtype}' for column '{col}'. Skipping conversion.")
                continue  # Skip if dtype is unknown

            try:
                if expected_dtype == "datetime64[ns]":
                    #Explicitly handle date conversion with format specified
                    df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce') 
                else:
                    df[col] = df[col].astype(expected_dtype)

            except (ValueError, TypeError) as e:
                logging.error(f"Failed to convert column '{col}' to dtype '{expected_dtype}'. Error: {e}")
                raise  Exception(f"Column {col} has incorrect dtype. Expected:{expected_dtype}, Found:{df[col].dtype}")
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validated_dir = self.data_validation_config.validated_dir
            validated_file_path = self.validate_and_compile_data()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path = self.data_validation_config.schema_file_path,
                validated_dir = validated_dir,
                validated_file_path = validated_file_path,
                is_validated = True,
                message = "Data Validation completed successfully"
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise vmException(e,sys) from e
