from collections import namedtuple

DataIngestionConfig = namedtuple('DataIngestionConfig', 
["raw_data_dir", "ingested_dir"])

DataValidationConfig = namedtuple('DataValidationConfig',
["schema_file_path", "validated_dir"])

DataTransformationConfig = namedtuple('DataTransformationConfig',
["transformed_train_dir", "transformed_test_dir", "test_size", "random_state", "preprocessor_dir"])

ModelTrainerConfig = namedtuple('ModelTrainerConfig',
["trained_model_dir", "modelc_config_file_path", "modelr_config_file_path", "test_size", "random_state", "modelc_performance_dir", "modelr_performance_dir"])

# ModelEvaluationConfig = namedtuple('ModelTrainerConfig', ["model_evaluation_file_path", "time_stamp"])

ModelEvaluatorConfig = namedtuple('ModelTrainerConfig', ["model_evaluation_file_path", "model_config_file_path"])

# ModelPusherConfig = namedtuple('ModelPusherConfig', ["export_dir_path"])

# TrainingPipelineConfig = namedtuple('TrainingPipelineConfig', ["artifact_dir"])
