from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import ModelEvaluatorConfig
from vmpred.entity.artifactEntity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluatorArtifact
from vmpred.constant import *
from vmpred.util.util import read_yaml_file, read_parquet, load_object, save_object
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import pandas as pd
import numpy as np
import os, sys, joblib


class ModelEvaluator:
    def __init__(self, modelEvaluatorConfig: ModelEvaluatorConfig, dataTransformationArtifact=DataTransformationArtifact, modelTrainerArtifact=ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*30}Model Evaluation Log Started{'<<'*30} \n\n")
            self.modelEvaluationConfig = modelEvaluatorConfig
            self.data_transformation_artifact = dataTransformationArtifact
            self.modelTrainerArtifact = modelTrainerArtifact

        except Exception as e:
            raise vmException(e, sys) from e

    def initiate_model_evaluator(self) -> ModelEvaluatorArtifact:
        
        try:

            model_config_path = self.modelEvaluationConfig.model_config_file_path
            model_configs = read_yaml_file(model_config_path)

            validation_df = read_parquet(r'C:/Users/Sahil Khan/Desktop/Vehicle_Maintenance_Prediction/data/transformedData/testData/test_data.parquet')

            # schema = read_yaml_file(self.dataValidationArtifact.schema_file_path)
            target_variable = TARGET_VARIABLE
            y_true = validation_df[target_variable]
            X = validation_df.drop(target_variable, axis=1)


            model_results = []
            for model_config in model_configs['models']:
                model_name = model_config['model_name']
                module_name, class_name = model_name.rsplit('.', 1)
                trained_model_path = os.path.join(self.modelTrainerArtifact.trained_model_file_path, f"{class_name}.pkl")  # Load model from specified path
                trained_model = load_object(trained_model_path)

                y_pred = trained_model.predict(X)
                try:
                    y_prob = trained_model.predict_proba(X)[:, 1]
                except AttributeError as e:  #Handles cases where predict_proba doesn't exist
                    logging.info(f"Model {model_name} doesn't have predict_proba, setting roc_auc to NaN")
                    y_prob = np.nan

                model_results.append({
                    'model_name': class_name,
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred),
                    'recall': recall_score(y_true, y_pred),
                    'f1_score': f1_score(y_true, y_pred),
                    'roc_auc': roc_auc_score(y_true, y_prob) if not np.isnan(y_prob).any() else np.nan,
                    'log_loss': log_loss(y_true, trained_model.predict_proba(X)) if hasattr(trained_model, 'predict_proba') else np.nan
                })

            model_evaluation_df = pd.DataFrame(model_results)
            csv_file_path = self.modelEvaluationConfig.model_evaluation_file_path
            os.makedirs(csv_file_path, exist_ok=True)
            model_evaluation_df.to_csv(os.path.join(csv_file_path, "model_eval_report.csv"), index=False)
            logging.info(f"Model evaluation saved to: {csv_file_path}")

            best_model_name = model_evaluation_df.loc[model_evaluation_df['f1_score'].idxmax(), 'model_name']
            best_model_path = os.path.join(self.modelTrainerArtifact.trained_model_file_path, f"{best_model_name}.pkl")
            best_model = joblib.load(best_model_path)


            # Save best model to the evaluated_file_path
            best_model_path_eval = os.path.join(self.modelEvaluationConfig.model_evaluation_file_path, f'{best_model_name}.pkl')
            os.makedirs(os.path.dirname(self.modelEvaluationConfig.model_evaluation_file_path), exist_ok=True)
            save_object(best_model_path_eval, best_model_path_eval)


            model_evaluation_artifact = ModelEvaluatorArtifact(
                is_evaluated=True,
                best_model_path=best_model_path,
                model_evaluation_report_path=csv_file_path,
                best_model_name=best_model_name
            )

            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise vmException(e, sys) from e

    def __del__(self):
        logging.info(f"{'<<'*20}Model Evaluation Log Completed{'>>'*20} \n\n")