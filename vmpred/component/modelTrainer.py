from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import ModelTrainerConfig
from vmpred.entity.artifactEntity import DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from vmpred.util.util import read_yaml_file, read_parquet, save_object
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
import importlib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
import os, sys, time


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
                    
                    start_time = time.time()  # Record training start time
                    grid_search.fit(X_train, y_train)
                    end_time = time.time()
                    training_time = end_time - start_time

                    model_current = grid_search.best_estimator_

                    # Cross-validation
                    skf = StratifiedKFold(n_splits=crossValidation, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model_current, X_train, y_train, cv=skf, scoring='accuracy')

                    start_time_pred = time.time()

                    y_pred = model_current.predict(X_test)
                    y_proba = model_current.predict_proba(X_test)

                    end_time_pred = time.time()
                    prediction_time = end_time_pred - start_time_pred

                    accuracy = accuracy_score(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    logloss = log_loss(y_test, y_proba)


                    try: # Use try-except block for roc_auc_score
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])  # Use probabilities for class 1
                    except ValueError as ve: # For multiclass problems where there is only one label present in test set
                        logging.warning(f"ValueError encountered during roc_auc_score calculation for model: {model_name}. Setting ROC_AUC to np.nan.  Error message: {ve}")
                        roc_auc = np.nan # Handle the case

                    results.append({
                    'model_name': class_name,
                    'best_params': grid_search.best_params_,
                    'accuracy': accuracy,
                    'cross_val_mean_accuracy': cv_scores.mean(),
                    'cross_val_std': cv_scores.std(),
                    'roc_auc': roc_auc,
                    'log_loss': logloss,
                    'confusion_matrix': conf_matrix.tolist(),
                    'precision': class_report['weighted avg']['precision'],
                    'recall': class_report['weighted avg']['recall'],
                    'f1_score': class_report['weighted avg']['f1-score'],
                    'training_time': training_time,
                    'prediction_time': prediction_time
                })
                    logging.info(f"""Training Model Result: 
                                    'model_name': {class_name},
                                    'best_params': {grid_search.best_params_},
                                    'accuracy': {accuracy},
                                    'cross_val_mean_accuracy': {cv_scores.mean()},
                                    'cross_val_std': {cv_scores.std()},
                                    'roc_auc': {roc_auc},
                                    'log_loss': {logloss},
                                    'confusion_matrix': {conf_matrix.tolist()},
                                    'precision': {class_report['weighted avg']['precision']},
                                    'recall': {class_report['weighted avg']['recall']},
                                    'f1_score': {class_report['weighted avg']['f1-score']},
                                    'training_time': {training_time},
                                    'prediction_time': {prediction_time}""")
                    
                    model_path = os.path.join(self.model_trainer_config.trained_model_dir, f'{class_name}.pkl')
                    save_object(model_path, model_current)
                    logging.info(f"Model: ({model_name}) saved to: {model_path}")
                    
                except Exception as e:
                    raise vmException(e,sys) from e
                
            results_df = pd.DataFrame(results)
            csv_file_path = os.path.join(self.model_trainer_config.model_performance_dir, 'model_performance.csv')
            results_df.to_csv(csv_file_path, index=False)
            

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message='Model Training Completed Successfully',
                trained_model_file_path=self.model_trainer_config.trained_model_dir
            )
            
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            raise vmException(e,sys) from e
        
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        
        try:
            
            # train_data_file_path = os.path.join(self.data_transformation_artifact.transformed_train_file_path, "train_data.parquet")

            processedData = read_parquet(self.data_transformation_artifact.transformed_train_file_path)

            modelsConfigurationInfo = self.model_trainer_config.model_config_file_path

            X_train, X_test, y_train, y_test = self.split_data(processedData, self.data_validation_artifact.schema_file_path)


            return self.train_model(X_train, X_test, y_train, y_test, modelsConfigurationInfo)

        except Exception as e:
            raise vmException(e,sys) from e




    def __del__(self):
        logging.info(f"{'>>'*20}Model Training Log Completed{'<<'*20}\n\n")