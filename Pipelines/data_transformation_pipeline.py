from utils.database_utils import get_table_names, connect
from utils.config_utils import get_db_config, read_config
from utils.feature_group_utils import hopswork_login, fetch_df_from_feature_groups
from constants import CONFIG_FILE_PATH
from components.data_transformation import DataTransformation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

STAGE_NAME = "Data Transformation"


class DataTransformationPipeline():
    def __init__(self):
        pass

    def main(self):
        project = hopswork_login()
        fs = project.get_feature_store()
        
        table_names = ['final_merge']

        df_dict = fetch_df_from_feature_groups(fs, table_names, 1)
        data_transformation_obj = DataTransformation()
        
        df_dict = data_transformation_obj.drop_event_time_column(
            df_dict)
        final_merge = df_dict['final_merge_fg']
        
        cts_cols=['route_avg_temp', 'route_avg_wind_speed',
              'route_avg_precip', 'route_avg_humidity', 'route_avg_visibility',
              'route_avg_pressure', 'distance', 'average_hours',
              'origin_temp', 'origin_wind_speed', 'origin_precip', 'origin_humidity',
              'origin_visibility', 'origin_pressure',
              'destination_temp','destination_wind_speed','destination_precip',
              'destination_humidity', 'destination_visibility','destination_pressure',
               'avg_no_of_vehicles', 'truck_age','load_capacity_pounds', 'mileage_mpg',
               'age', 'experience','average_speed_mph']
       
        cat_cols=['route_description',
                'origin_description', 'destination_description',
                'accident', 'fuel_type',
                'gender', 'driving_style', 'ratings','is_midnight']

        target=['delay']
        
        feature_cols = cts_cols + cat_cols
        
        train_df, validation_df, test_df = data_transformation_obj.split_data(final_merge, train_size=0.6, valid_size=0.2, test_size=0.2, random_state=42)
        
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data_transformation_obj.extract_features_and_target(train_df, validation_df, test_df, feature_cols, target)
        
        encoder_columns = ['route_description', 'origin_description', 'destination_description', 'fuel_type', 'gender', 'driving_style']
        X_train, X_valid, X_test = data_transformation_obj.onehotencode_and_transform(X_train, X_valid, X_test, encoder_columns, encoder_path='./models/encoders/onehotEncoder.pkl')
        
        
        label_encoder_columns = ['accident', 'ratings','is_midnight']
        X_train, X_valid, X_test = data_transformation_obj.label_encode_and_transform(X_train, X_valid, X_test, label_encoder_columns, encoder_dir='./models/encoders/')
        
        X_train, X_valid, X_test = data_transformation_obj.drop_encoded_columns(X_train, X_valid, X_test, encoder_columns)
        
        X_train, X_valid, X_test = data_transformation_obj.scale_features(X_train, X_valid, X_test, cts_cols, scaler_path='./models/encoders/standardScaler.pkl')
        
        data_transformation_obj.initialize_experiment()
        model_registry = data_transformation_obj.get_hopsworks_registory(project)
        
        # Get hyperparameter grids
        # param_grids = data_transformation_obj.get_hyperparameter_grids()
        
        # # Train, tune, and log each model
        # models = {
        #     'Logistic_Regression': LogisticRegression(),
        #     'Random_Forest': RandomForestClassifier(),
        #     'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        # }
        
        # for model_name, model in models.items():
        #     param_grid = param_grids[model_name]
        #     data_transformation_obj.train_and_log_model(model, param_grid, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, model_registry)
        
        data_transformation_obj.save_encoders(model_registry)
        
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :", STAGE_NAME)
        obj = DataTransformationPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e
        
        