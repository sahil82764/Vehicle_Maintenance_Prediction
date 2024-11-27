from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import DataTransformationConfig
from vmpred.entity.artifactEntity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
import sys, os, io
import numpy as np
import pandas as pd
from vmpred.constant import *
from vmpred.util.util import read_parquet, read_yaml_file
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

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
            logging.info(f"""Missing Values: \n {df.isnull().sum()}""")
            return df.dropna()
        except Exception as e:
            raise vmException(e,sys) from e
        
    def drop_duplicates(self, df: pd.DataFrame):
        try:
            logging.info("Duplicates Dropped if any")
            return df.drop_duplicates()
        except Exception as e:
            raise vmException(e,sys) from e
        
    def feature_enginnering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            
            # 1. Time Since Last Service (in days)
            df['time_since_last_service'] = (REFERENCE_DATE - df['last_service_date']).dt.days

            # 2. Warranty Duration (in days)
            df['warranty_duration'] = (df['warranty_expiry_date'] - REFERENCE_DATE).dt.days

            # 3. Mileage per Year
            df['mileage_per_year'] = df['mileage'] / df['vehicle_age']

            # 4. Service Frequency
            df['service_frequency'] = df['service_history'] / df['vehicle_age']

            # 5. Accident Rate
            df['accident_rate'] = df['accident_history'] / df['vehicle_age']

            # 6. Ordinal Encoding of Maintenance History
            maintenance_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
            df['maintenance_history'] = df['maintenance_history'].map(maintenance_mapping)

            # 7. Ordinal Encoding of Tire and Brake Condition
            condition_mapping = {'Worn Out': 1, 'Good': 2, 'New': 3}
            df['tire_condition'] = df['tire_condition'].map(condition_mapping)
            df['brake_condition'] = df['brake_condition'].map(condition_mapping)

            # 8. Ordinal Encoding of Battery Condition
            status_mapping = {'Weak': 1, 'Good': 2, 'New': 3}
            df['battery_status'] = df['battery_status'].map(status_mapping)

            buffer = io.StringIO()  # Create a buffer to capture the output
            df.info(buf=buffer)  # Write the DataFrame info to the buffer
            info_str = buffer.getvalue()  # Get the string content of the buffer

            logging.info(f"New Features Added: \n {info_str}")

            return df

        except Exception as e:
            raise vmException(e,sys) from e
        
    def seperate_and_scale(self, df: pd.DataFrame, file_path: str):
        try:            
            
            y = df[[TARGET_VARIABLE[0],TARGET_VARIABLE[1]]]
            X = df.drop(columns=TARGET_VARIABLE, axis=1)

            # Identify numerical and categorical features
            numerical_cols = X.select_dtypes(include=np.number).columns
            categorical_cols = X.select_dtypes(include=['category']).columns

            # Create a column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ])
            
            # Apply transformations
            X_processed = preprocessor.fit_transform(X)
            
            #Get feature names for OneHotEncoder
            encoded_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

            #Combine feature names
            feature_names = list(numerical_cols) + encoded_feature_names

            #Create DataFrame with preserved column names
            X_processed = pd.DataFrame(X_processed, columns=feature_names)

            # Recombine features and target
            X_processed[TARGET_VARIABLE] = y

            self.preprocessor_path = os.path.join(self.data_transformation_config.preprocessor_dir,"preprocessor.pkl")
            joblib.dump(preprocessor, self.preprocessor_path)  # Save the preprocessor
            logging.info(f"Preprocessor saved to: {self.preprocessor_path}")

            logging.info("Feature Scaling [Numerical Cols] and One-Hot Encoding [Categorical Columns] Completed")
            logging.info(f"Dimension of Processed Data: {X_processed.shape}")

            return X_processed


        except Exception as e:
            raise vmException(e,sys) from e
        
    def split_data_train_test(self, df: pd.DataFrame) -> DataTransformationArtifact:

        try:
            train_dir = self.data_transformation_config.transformed_train_dir
            test_dir = self.data_transformation_config.transformed_test_dir

            testSize = self.data_transformation_config.test_size
            randomState = self.data_transformation_config.random_state

            logging.info("Splitting Data into Train and Test")

            split = StratifiedShuffleSplit(n_splits=1, test_size=testSize, random_state=randomState)

            for train_index, test_index in split.split(df, df[TARGET_VARIABLE[0]]):
                strat_train_set = df.iloc[train_index]
                strat_test_set = df.iloc[test_index]

            logging.info(f"Exporting training dataset to file: [{train_dir}]")
            parquet_file_train_path = os.path.join(train_dir, "train_data.parquet")
            strat_train_set.to_parquet(parquet_file_train_path, engine="pyarrow")

            logging.info(f"Exporting testing dataset to file: [{test_dir}]")
            parquet_file_test_path = os.path.join(test_dir, "test_data.parquet")
            strat_test_set.to_parquet(parquet_file_test_path, engine="pyarrow")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=parquet_file_train_path,
                transformed_test_file_path=parquet_file_test_path,
                is_transformed=True,
                message="Data Transformation completed successfully",
                preprocessor_file_path=self.preprocessor_path
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
  

        except Exception as e:
            raise vmException(e,sys) from e
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        
        try:
            validated_file_path = self.data_validation_artifact.validated_file_path
            vmData = read_parquet(validated_file_path)

            vmData = self.handle_missing_values(vmData)
            vmData = self.drop_duplicates(vmData)
            vmData = self.feature_enginnering(vmData)
            vmData = self.seperate_and_scale(vmData, self.data_validation_artifact.schema_file_path)
            return self.split_data_train_test(vmData)

        except Exception as e:
            raise vmException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*20}Data Transforamtion Log Completed{'<<'*20}\n\n")