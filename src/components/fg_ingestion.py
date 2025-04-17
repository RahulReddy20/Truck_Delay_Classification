import hopsworks
import pandas as pd
import time
from datetime import datetime
from sqlalchemy import create_engine, MetaData
from constants import *
import warnings
from utils.config_utils import *
from utils.database_utils import *
from utils.feature_group_utils import create_feature_groups


class fgIngestion:
    def __init__(self):
        self.config = read_config(CONFIG_FILE_PATH)
        self.api_key = self.config['API']['hopswork_api_key']
        self.db_config = get_db_config(self.config)
        self.engine = connect(self.db_config)
        self.dataframes_dict = read_all_tables(self.engine)

    def hopswork_login(self):
        return hopsworks.login(api_key_value=self.api_key)

    def add_id_and_event_time_columns(self):
        today_date = datetime.now().date()
        for table_name, df in self.dataframes_dict.items():
            if 'id' not in df.columns:
                df.insert(0, 'id', range(1, len(df) + 1)) 
                print(f"'id' column added for table: {table_name}")
            else:
                print(f"'id' column already exists in table: {table_name}")

            if 'event_time' not in df.columns:
                df['event_time'] = [today_date] * \
                    len(df)  
                print(f"'event_time' column added for table: {table_name}")
            else:
                print(
                    f"'event_time' column already exists in table: {table_name}")

            self.dataframes_dict[table_name] = df

    def fill_missing_string_values(self):
        for table_name, df in self.dataframes_dict.items():
            string_columns = df.select_dtypes(include=['object'])
            missing_values = string_columns.isna().sum()
            missing_columns = missing_values[missing_values > 0]

            if not missing_columns.empty:
                print(
                    f"Table: {table_name} has missing values in columns: {missing_columns.index.tolist()}")
                for col in missing_columns.index:
                    # Get mode (most frequent value)
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                    print(
                        f"Filled missing values in column '{col}' with mode '{mode_value}'")

                self.dataframes_dict[table_name] = df

    def convert_string_to_datetime(self):
        warnings.simplefilter(action='ignore', category=UserWarning)

        for table_name, df in self.dataframes_dict.items():
            object_columns = df.select_dtypes(include=['object']).columns
            for col in object_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(
                        f"Converted '{col}' to datetime in table '{table_name}'")
                except (ValueError, TypeError):
                    print(
                        f"Column '{col}' in table '{table_name}' is not a valid date format.")
            self.dataframes_dict[table_name] = df

    def process_feature_groups(self, fs, ver, dataframes=None, batch_size=5):
        if dataframes is None:
            dataframes = self.dataframes_dict

        create_feature_groups(fs, ver, dataframes, batch_size)

    def save_dataframes_to_db(self, if_exists='replace'):
        if self.engine is None:
            print("error connecting to database")
        for table_name, df in self.dataframes_dict.items():
            try:
                df.to_sql(name=table_name, con=self.engine,
                          if_exists=if_exists, index=False)
                print(
                    f"Table '{table_name}'_cleaned successfully saved to the database.")
            except Exception as e:
                print(f"Error saving table '{table_name}_cleaned': {e}")
