from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import ModelTrainerConfig
from vmpred.entity.artifactEntity import DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from vmpred.util.util import read_yaml_file, read_parquet, save_object
from sklearn.model_selection import GridSearchCV, train_test_split
import importlib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os, sys


class ModelTrainer:
    
    def __init__(self, modelTrainerConfig=ModelTrainerConfig, dataTransformationArtifact=DataTransformationArtifact, dataValidationArtifact=DataValidationArtifact):
        try:
            logging.info(f"{'>>'*30}Model Training log started{'<<'*30}\n\n")
            
            self.model_trainer_config = modelTrainerConfig
            self.data_transformation_artifact = dataTransformationArtifact
            self.data_validation_artifact = dataValidationArtifact

        except Exception as e:
            raise vmException(e,sys) from e
        
    def split_data(self, df: pd.DataFrame, file_path: str):
        
        try:

            schema = read_yaml_file(file_path=file_path)
            
            TARGET_VARIABLE = schema['target_column'][0]
            
            y = df[TARGET_VARIABLE]
            X = df.drop(TARGET_VARIABLE, axis=1)

            testSize = self.model_trainer_config.test_size
            randomState = self.model_trainer_config.random_state
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
            
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise vmException(e,sys) from e
        

    def train_model(self, X_train, X_test, y_train, y_test, model_config_path: str) -> ModelTrainerArtifact:
        try:
            
            model_configs = read_yaml_file(file_path=model_config_path)

            crossValidation = model_configs.get('cross_validation')

            results = []
            best_model = None
            best_model_name = None
            base_accuracy = self.model_trainer_config.base_accuracy

            for model_config in model_configs['models']:
                model_name = model_config['model_name']
                hyperparameters = model_config['hyperparameters']

                try:
                    
                    module_name, class_name = model_name.rsplit('.', 1)
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, class_name)
                    model = model_class()
                    logging.info(f'Training Model:{class_name}')

                    grid_search = GridSearchCV(model, hyperparameters, cv=crossValidation, scoring='accuracy', n_jobs=-1, error_score='raise', verbose=1)
                    grid_search.fit(X_train, y_train)

                    best_model_current = grid_search.best_estimator_
                    y_pred = best_model_current.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, output_dict=True)

                    results.append({
                        'model_name': model_name,
                        'best_params': grid_search.best_params_,
                        'accuracy': accuracy,
                        'confusion_matrix': conf_matrix.tolist(),
                        'precision': class_report['weighted avg']['precision'],
                        'recall': class_report['weighted avg']['recall'],
                        'f1_score': class_report['weighted avg']['f1-score']
                    })
                    logging.info(f"""Training Model Result: 
                                'model_name': {model_name},
                                'best_params': {grid_search.best_params_},
                                'accuracy': {accuracy},
                                'confusion_matrix': {conf_matrix.tolist()},
                                'precision': {class_report['weighted avg']['precision']},
                                'recall': {class_report['weighted avg']['recall']},
                                'f1_score': {class_report['weighted avg']['f1-score']}""")

                    if accuracy > base_accuracy:
                        base_accuracy = accuracy
                        best_model = best_model_current
                        best_model_name = model_name
                    
                except Exception as e:
                    raise vmException(e,sys) from e
                
            results_df = pd.DataFrame(results)
            csv_file_path = os.path.join(self.model_trainer_config.model_performance_dir, 'model_performance.csv')
            results_df.to_csv(csv_file_path, index=False)

            if best_model is not None:
                best_model_path = os.path.join(self.model_trainer_config.trained_model_dir, f'{best_model_name}_best_model.pkl')
                save_object(best_model_path, best_model)
                logging.info(f"Best model ({best_model_name}) saved to: {best_model_path}")

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message='Model Training Completed Successfully',
                trained_model_file_path=best_model_path,
                model_name= best_model_name,
                best_accuracy = base_accuracy,
                precision=class_report['weighted avg']['precision'],
                recall=class_report['weighted avg']['recall'],
                f1_score=class_report['weighted avg']['f1-score']                
            )
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            raise vmException(e,sys) from e
        
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        
        try:
            
            train_data_file_path = os.path.join(self.data_transformation_artifact.transformed_train_file_path, "train_data.parquet")

            processedData = read_parquet(train_data_file_path)

            modelsConfigurationInfo = self.model_trainer_config.model_config_file_path

            X_train, X_test, y_train, y_test = self.split_data(processedData, self.data_validation_artifact.schema_file_path)


            return self.train_model(X_train, X_test, y_train, y_test, modelsConfigurationInfo)

        except Exception as e:
            raise vmException(e,sys) from e




    def __del__(self):
        logging.info(f"{'>>'*20}Model Training Log Completed{'<<'*20}\n\n")