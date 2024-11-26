from collections import namedtuple

DataIngestionArtifact = namedtuple('DataIngestionArtifact',
["ingested_data_dir", "is_ingested", "message"])

DataValidationArtifact = namedtuple('DataValidationArtifact',
["schema_file_path", "validated_dir", "validated_file_path", "is_validated", "message"])

DataTransformationArtifact = namedtuple('DataTransformationArtifact',
["transformed_train_file_path", "transformed_test_file_path", "is_transformed", "message", "preprocessor_file_path"])

# ModelTrainerArtifact = namedtuple('ModelTrainerArtifact',
# ["is_trained", "message", "trained_model_file_path", "model_name", "best_accuracy", "precision", "recall", "f1_score"])

ModelTrainerArtifact = namedtuple('ModelTrainerArtifact',
["is_trained", "message", "trained_model_file_path"])

ModelEvaluatorArtifact = namedtuple('ModelEvaluatorArtifact',
["is_evaluated","best_model_path", "model_evaluation_report_path", "best_model_name"])

# ModelPusherArtifact = namedtuple('ModelPusherArtifact',
# ["is_model_pusher", "export_model_file_path"])