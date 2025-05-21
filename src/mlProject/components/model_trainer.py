import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.Preprocesiing.preprocesing import encode_categorical



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            "R2 Score": r2
        }

        logger.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

    def train(self):
        try:
            logger.info("Loading training and test data")
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            logger.info("Encoding categorical features")
            train_data = encode_categorical(train_data)
            test_data = encode_categorical(test_data)

            logger.info("Splitting data into features and target")
            X_train = train_data.drop(self.config.target_column, axis=1)
            y_train = train_data[self.config.target_column]
            X_test = test_data.drop(self.config.target_column, axis=1)
            y_test = test_data[self.config.target_column]

            logger.info("Initializing Random Forest model")
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                bootstrap=self.config.bootstrap,
                random_state=self.config.random_state,
                verbose=1
            )

            logger.info("Training model...")
            model.fit(X_train, y_train)
            logger.info("Model training completed")

            logger.info("Evaluating model performance")
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            logger.info("Training set metrics:")
            self.evaluate_model(y_train, train_pred)

            logger.info("Test set metrics:")
            test_metrics = self.evaluate_model(y_test, test_pred)

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(model, model_path)
            logger.info(f"Model saved at: {model_path}")

            return test_metrics

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise


# Main execution
if __name__ == "__main__":
    try:
        logger.info("Starting model training pipeline")

        from mlProject.configuration.configuration import ConfigurationManager
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()

        model_trainer = ModelTrainer(config=model_trainer_config)
        metrics = model_trainer.train()

        logger.info(f"Model training completed successfully with metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise
