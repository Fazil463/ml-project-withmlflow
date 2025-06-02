from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
import pandas as pd
import os
from mlProject import logger
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        try:
            if hasattr(self.params, 'RandomForest'):
                params = self.params.RandomForest
            elif hasattr(self.params, 'random_forest'):
                params = self.params.random_forest
            elif hasattr(self.params, 'randomforest'):
                params = self.params.randomforest
            else:
                raise ValueError("RandomForest parameters not found in params.yaml")

            schema = self.schema.TARGET_COLUMN

            if not os.path.exists(config.train_data_path):
                raise FileNotFoundError(f"Train data path not found: {config.train_data_path}")
            if not os.path.exists(config.test_data_path):
                raise FileNotFoundError(f"Test data path not found: {config.test_data_path}")

            create_directories([config.root_dir])

            return ModelTrainerConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                target_column=schema,
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_samples_split=params.min_samples_split,
                min_samples_leaf=params.min_samples_leaf,
                max_features=params.max_features,
                bootstrap=params.bootstrap,
                random_state=params.random_state
            )
        except Exception as e:
            logger.error(f"Error in getting model trainer config: {e}")
            raise

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            config = self.config.model_evaluation
            schema = self.schema.TARGET_COLUMN  # This is already a string

            # Safely extract RandomForest params
            rf_keys = ["RandomForest", "random_forest", "randomforest"]
            params = None
            for key in rf_keys:
                if hasattr(self.params, key):
                    params = getattr(self.params, key)
                    break
            if params is None:
                raise ValueError("RandomForest parameters not found in params.yaml")

            # Ensure model evaluation directory exists
            create_directories([config.root_dir])

            return ModelEvaluationConfig(
                root_dir=config.root_dir,
                test_data_path=config.test_data_path,
                model_path=config.model_path,
                all_params=params,
                metric_file_name=config.metric_file_name,
                target_column=schema,  # ✅ FIXED HERE
                mlflow_uri="https://dagshub.com/fazilkkv123/ml-project-withmlflow.mlflow"  # Update with your actual URI
            )

        except Exception as e:
            logger.error(f"Error in getting model evaluation config: {e}")
            raise 