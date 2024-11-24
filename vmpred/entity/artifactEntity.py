from collections import namedtuple

DataIngestionArtifact = namedtuple('DataIngestionArtifact',
["ingested_data_dir", "is_ingested", "message"])

DataValidationArtifact = namedtuple('DataValidationArtifact',
["schema_file_path", "validated_dir", "validated_file_path", "is_validated", "message"])

DataTransformationArtifact = namedtuple('DataTransformationArtifact',
["transformed_train_file_path", "transformed_test_file_path","is_transformed", "message",])

ModelTrainerArtifact = namedtuple('ModelTrainerArtifact',
["is_trained", "message", "trained_model_file_path", "model_name", "best_accuracy", "precision", "recall", "f1_score"])

# ModelEvaluationArtifact = namedtuple('ModelEvaluationArtifact',
# ["is_model_accepted","evaluated_model_path"])

# ModelPusherArtifact = namedtuple('ModelPusherArtifact',
# ["is_model_pusher", "export_model_file_path"])