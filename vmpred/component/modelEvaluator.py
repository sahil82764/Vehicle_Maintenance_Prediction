from vmpred.exception import vmException
from vmpred.logger import logging
from vmpred.entity.configEntity import ModelEvaluatorConfig
from vmpred.entity.artifactEntity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluatorArtifact
from vmpred.constant import *
from vmpred.util.util import read_yaml_file, read_parquet, load_object, save_object
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os, sys, joblib


class ModelEvaluator:
    def __init__(self, modelEvaluatorConfig: ModelEvaluatorConfig, dataTransformationArtifact: DataTransformationArtifact, modelTrainerArtifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*30}Model Evaluation Log Started{'<<'*30} \n\n")
            self.modelEvaluationConfig = modelEvaluatorConfig
            self.dataTransformationArtifact = dataTransformationArtifact
            self.modelTrainerArtifact = modelTrainerArtifact

        except Exception as e:
            raise vmException(e, sys) from e

    def evaluate_classification(self, y_true, y_pred, y_prob):
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_prob) if not np.isnan(y_prob).any() else np.nan
            logloss = log_loss(y_true, y_prob)
            return accuracy, precision, recall, f1, roc_auc, logloss
        except Exception as e:
            raise vmException(e, sys) from e

    def evaluate_regression(self, y_true, y_pred):
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return rmse, mae, r2
        except Exception as e:
            raise vmException(e, sys) from e

    def get_best_model_c(self, model_evaluation_df):
        try:
            best_model_name = None
            if model_evaluation_df.shape[0] > 0:
                model_evaluation_df = model_evaluation_df.sort_values(by=['f1_score'], ascending=[False])
                best_model_name = model_evaluation_df.iloc[0].model_name
            else:
                raise ValueError("no model evaluation data found.")
            return best_model_name
        except Exception as e:
            raise vmException(e, sys) from e
    
    def get_best_model_r(self, model_evaluation_df):
        try:
            best_model_name = None
            if model_evaluation_df.shape[0] > 0:
                model_evaluation_df = model_evaluation_df.sort_values(by=['rmse'], ascending=[True])
                best_model_name = model_evaluation_df.iloc[0].model_name
            else:
                raise ValueError("no model evaluation data found.")
            return best_model_name
        except Exception as e:
            raise vmException(e, sys) from e

    def initiate_model_evaluator(self) -> ModelEvaluatorArtifact:
        try:
            model_config_path = self.modelEvaluationConfig.model_config_file_path
            model_configs = read_yaml_file(model_config_path)
            validation_df = read_parquet(self.dataTransformationArtifact.transformed_test_file_path)

            target_variable = TARGET_VARIABLE
            y_true_c = validation_df[target_variable[0]]
            y_true_r = validation_df[target_variable[1]]
            X = validation_df.drop(columns=target_variable, axis=1)


            model_results = []
            for model_config in model_configs['models']:
                model_name = model_config['model_name']
                class_name = model_name.split('.')[-1]
                trained_model_path = os.path.join(self.modelTrainerArtifact.trained_model_file_path, f"{class_name}.pkl")
                trained_model = load_object(trained_model_path)

                y_pred = trained_model.predict(X)

                if model_name.split('.')[-1] in ['LogisticRegression', 'RandomForestClassifier', 'SGDClassifier', 'DecisionTreeClassifier', 'XGBClassifier']:
                    y_prob = trained_model.predict_proba(X)[:, 1] if hasattr(trained_model, 'predict_proba') else np.nan
                    accuracy, precision, recall, f1, roc_auc, logloss = self.evaluate_classification(y_true_c, y_pred, y_prob)
                    model_results.append({
                        'model_name': class_name,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'log_loss': logloss
                    })

                elif model_name.split('.')[-1] in ['LinearRegression', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor']:
                    rmse, mae, r2 = self.evaluate_regression(y_true_r, y_pred)
                    model_results.append({
                        'model_name': class_name,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    })
                else:
                    raise ValueError(f"Unsupported model type: {model_name}")

            model_evaluation_df = pd.DataFrame(model_results)
            model_eval_report_path = os.path.join(self.modelEvaluationConfig.model_evaluation_file_path, "model_eval_report.csv")
            os.makedirs(self.modelEvaluationConfig.model_evaluation_file_path, exist_ok=True)
            model_evaluation_df.to_csv(model_eval_report_path, index=False)
            logging.info(f"Model evaluation saved to: {model_eval_report_path}")

            best_model_name_c = self.get_best_model_c(model_evaluation_df)
            best_model_name_r = self.get_best_model_r(model_evaluation_df)

            best_model_path_c = os.path.join(self.modelTrainerArtifact.trained_model_file_path, f"{best_model_name_c}.pkl")
            best_model_path_r = os.path.join(self.modelTrainerArtifact.trained_model_file_path, f"{best_model_name_r}.pkl")

            best_model_c = joblib.load(best_model_path_c)
            best_model_r = joblib.load(best_model_path_r)

            best_model_path_eval_c = os.path.join(self.modelEvaluationConfig.model_evaluation_file_path, f"{best_model_name_c}.pkl")
            best_model_path_eval_r = os.path.join(self.modelEvaluationConfig.model_evaluation_file_path, f"{best_model_name_r}.pkl")

            os.makedirs(self.modelEvaluationConfig.model_evaluation_file_path, exist_ok=True)
            save_object(best_model_path_eval_c, best_model_c)
            save_object(best_model_path_eval_r, best_model_r)

            model_evaluation_artifact = ModelEvaluatorArtifact(
                is_evaluated=True,
                best_model_path_c=best_model_path_eval_c,
                best_model_path_r=best_model_path_eval_r,
                model_evaluation_report_path=model_eval_report_path,
                best_model_name_c=best_model_name_c,
                best_model_name_r=best_model_name_r
            )

            logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise vmException(e, sys) from e

    def __del__(self):
        logging.info(f"{'<<'*20}Model Evaluation Log Completed{'>>'*20} \n\n")