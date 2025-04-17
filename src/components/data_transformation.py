import numpy as np
import pandas as pd
from utils.feature_group_utils import fetch_df_from_feature_groups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import joblib
import os
from datetime import datetime
import hsml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class DataTransformation():
    def __init__(self):
        self.model_dir = './models/'
    
    def drop_event_time_column(self, feature_dataframes):
        for key, df in feature_dataframes.items():
            if 'event_time' in df.columns:
                df.drop(columns=['event_time'], inplace=True)
        return feature_dataframes
    
    def split_data(self, df, train_size=0.6, valid_size=0.2, test_size=0.2, random_state=42):
        if not (train_size + valid_size + test_size) == 1.0:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        train_df, temp_df = train_test_split(df, test_size=(1 - train_size), random_state=random_state, shuffle=True)

        validation_df, test_df = train_test_split(temp_df, test_size=(test_size / (valid_size + test_size)), random_state=random_state, shuffle=True)

        return train_df, validation_df, test_df
    
    def extract_features_and_target(self, train_df, validation_df, test_df, feature_cols, target_col):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_valid = validation_df[feature_cols]
        y_valid = validation_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def onehotencode_and_transform(self, X_train, X_valid, X_test, encoder_columns, encoder_path):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(X_train[encoder_columns])
        
        encoded_features = list(encoder.get_feature_names_out(encoder_columns))
        
        X_train_encoded = encoder.transform(X_train[encoder_columns])
        X_valid_encoded = encoder.transform(X_valid[encoder_columns])
        X_test_encoded = encoder.transform(X_test[encoder_columns])
        
        X_train[encoded_features] = X_train_encoded
        X_valid[encoded_features] = X_valid_encoded
        X_test[encoded_features] = X_test_encoded
        
        joblib.dump(encoder, encoder_path)
        
        return X_train, X_valid, X_test
    
    def label_encode_and_transform(self, X_train, X_valid, X_test, label_encoder_columns, encoder_dir):
        
        if not os.path.exists(encoder_dir):
            os.makedirs(encoder_dir)

        for col in label_encoder_columns:
            encoder = LabelEncoder()
            X_train[col] = encoder.fit_transform(X_train[col])
            X_valid[col] = encoder.transform(X_valid[col])
            X_test[col] = encoder.transform(X_test[col])
            
            encoder_path = os.path.join(encoder_dir, f'{col}_label_encoder.pkl')
            joblib.dump(encoder, encoder_path)

        return X_train, X_valid, X_test
    
    def drop_encoded_columns(self, X_train, X_valid, X_test, columns_to_drop):
        X_train = X_train.drop(columns=columns_to_drop, axis=1)
        X_valid = X_valid.drop(columns=columns_to_drop, axis=1)
        X_test = X_test.drop(columns=columns_to_drop, axis=1)
        
        return X_train, X_valid, X_test
    
    def scale_features(self, X_train, X_valid, X_test, cts_cols, scaler_path):
        scaler = StandardScaler()

        X_train[cts_cols] = scaler.fit_transform(X_train[cts_cols])
        X_valid[cts_cols] = scaler.transform(X_valid[cts_cols])
        X_test[cts_cols] = scaler.transform(X_test[cts_cols])

        joblib.dump(scaler, scaler_path)
        
        return X_train, X_valid, X_test
    
    def initialize_experiment(self):
        mlflow.set_experiment("ML Models with Hyperparameter Tuning")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def get_hopsworks_registory(self, project):
        return project.get_model_registry()

    def perform_grid_search(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def train_and_log_model(self, model, param_grid, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, model_registry):
        best_model, best_params = self.perform_grid_search(model, param_grid, X_train, y_train)
        
        with mlflow.start_run(run_name=model_name):
            y_valid_pred = best_model.predict(X_valid)
            acc = accuracy_score(y_valid, y_valid_pred)
            f1 = f1_score(y_valid, y_valid_pred, average='weighted')
            
            y_test_pred = best_model.predict(X_test)
            acc_test = accuracy_score(y_test, y_test_pred)

            mlflow.log_params(best_params)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "test_accuracy": acc_test})
            mlflow.sklearn.log_model(best_model, model_name)

            local_model_path = f"{self.model_dir}{model_name}_model.pkl"
            joblib.dump(best_model, local_model_path)

            model_instance = model_registry.python.create_model(
                name=model_name,
                metrics={"accuracy": acc, "f1_score": f1},
                description=f"{model_name} with hyperparameter tuning"
            )
            model_instance.save(local_model_path)

            print(f"Best Parameters for {model_name}: {best_params}")
            print(f"Validation Accuracy: {acc}, Validation F1 Score: {f1}")
            print(f"Test Accuracy: {acc_test}")
            print(classification_report(y_valid, y_valid_pred))

    def get_hyperparameter_grids(self):
        logreg_params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200, 500]
        }
        rf_params = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        xgb_params = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 10],
            'colsample_bytree': [0.3, 0.7, 1],
            'subsample': [0.8, 1],
        }
        return {'Logistic_Regression': logreg_params, 'Random_Forest': rf_params, 'XGBoost': xgb_params}
    
    def save_encoders(self, model_registry, encoder_dir="./models/encoders/"):
        # os.makedirs(encoder_dir, exist_ok=True)

        for file_name in os.listdir(encoder_dir):
            if file_name.endswith(".pkl"):
                local_encoder_path = os.path.join(encoder_dir, file_name)
                encoder_name = file_name.replace(".pkl", "")

                with mlflow.start_run(run_name=f"{encoder_name}"):
                    mlflow.log_artifact(local_encoder_path, artifact_path="encoders")

                encoder_instance = model_registry.python.create_model(
                    name=encoder_name,
                    description=f"{encoder_name} for categorical encoding"
                )
                encoder_instance.save(local_encoder_path)
                print(f"{encoder_name} saved to Hopsworks Model Registry.")