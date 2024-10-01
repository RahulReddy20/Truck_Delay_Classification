from utils.database_utils import get_table_names, connect
from utils.config_utils import get_db_config, read_config
from utils.feature_group_utils import hopswork_login, fetch_df_from_feature_groups, create_feature_groups
from constants import CONFIG_FILE_PATH
from components.data_cleaning import DataCleaning

STAGE_NAME = "Data Claning"

class DataCleaningPipeline():
    def __init__(self):
        pass
    
    def main(self):
        engine = connect(get_db_config(read_config(CONFIG_FILE_PATH)))
        table_names = get_table_names(engine)
        
        project = hopswork_login()
        fs = project.get_feature_store()
        
        df_dict = fetch_df_from_feature_groups(fs, table_names, 1)
        data_cleaning_obj = DataCleaning()
        
        integer_columns = {
            'city_weather_fg': ['temp', 'wind_speed', 'humidity', 'pressure'],
            'trucks_table_fg': ['truck_age', 'load_capacity_pounds', 'mileage_mpg'],
            'traffic_table_fg': ['no_of_vehicles'],
            'drivers_table_fg': ['age', 'experience', 'ratings', 'average_speed_mph'],
            'routes_table_fg': ['distance', 'average_hours'],
            'routes_weather_fg': ['temp', 'wind_speed', 'humidity', 'pressure']
        }
        
        data_cleaning_obj.fill_missing_with_mean(df_dict['trucks_table_fg'], 'load_capacity_pounds')
        data_cleaning_obj.fill_missing_with_mean(df_dict['traffic_table_fg'], 'no_of_vehicles')
        
        df_dict['drivers_table_fg'] = data_cleaning_obj.drop_subset_rows(df_dict['drivers_table_fg'], df_dict['drivers_table_fg'][df_dict['drivers_table_fg']['experience']<0])
        
        df_dict['traffic_table_fg']['time_period'] = df_dict['traffic_table_fg']['hour'].apply(data_cleaning_obj.categorize_time_of_day)
        
        for table_name, columns in integer_columns.items():
            df = df_dict[table_name]
            outliers_df = data_cleaning_obj.iqr_outliers(df, columns)
            df_dict[table_name] = data_cleaning_obj.drop_subset_rows(df, outliers_df)
            
        print(df_dict)
        
        create_feature_groups(fs, 2, df_dict)
            
        
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = DataCleaningPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e