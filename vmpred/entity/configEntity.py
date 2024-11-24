from collections import namedtuple

DataIngestionConfig = namedtuple('DataIngestionConfig', 
["raw_data_dir", "ingested_dir"])

DataValidationConfig = namedtuple('DataValidationConfig',
["schema_file_path", "validated_dir"])

DataTransformationConfig = namedtuple('DataTransformationConfig',
["transformed_train_dir", "transformed_test_dir", "test_size", "random_state"])

ModelTrainerConfig = namedtuple('ModelTrainerConfig',
["trained_model_dir", "base_accuracy", "model_config_file_path", "test_size", "random_state", "model_performance_dir"])

# ModelEvaluationConfig = namedtuple('ModelTrainerConfig', ["model_evaluation_file_path", "time_stamp"])

# ModelPusherConfig = namedtuple('ModelPusherConfig', ["export_dir_path"])

# TrainingPipelineConfig = namedtuple('TrainingPipelineConfig', ["artifact_dir"])


