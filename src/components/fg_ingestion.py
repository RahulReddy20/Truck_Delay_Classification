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
        return hopsworks.login(api_key_value = self.api_key)

    def add_id_and_event_time_columns(self):
        """
        Add 'id' and 'event_time' columns to each DataFrame in the dictionary.
        """
        today_date = datetime.now().date()
        for table_name, df in self.dataframes_dict.items():
            # Check if 'id' column exists
            if 'id' not in df.columns:
                df.insert(0, 'id', range(1, len(df) + 1))  # Insert 'id' column
                print(f"'id' column added for table: {table_name}")
            else:
                print(f"'id' column already exists in table: {table_name}")
            
            # Check if 'event_time' column exists
            if 'event_time' not in df.columns:
                df['event_time'] = [today_date] * len(df)  # Add 'event_time' column
                print(f"'event_time' column added for table: {table_name}")
            else:
                print(f"'event_time' column already exists in table: {table_name}")
                
            # Update the DataFrame in the dictionary
            self.dataframes_dict[table_name] = df

    def fill_missing_string_values(self):
        """
        Fill missing values in string columns with the mode value.
        """
        for table_name, df in self.dataframes_dict.items():
            string_columns = df.select_dtypes(include=['object'])
            missing_values = string_columns.isna().sum()
            missing_columns = missing_values[missing_values > 0]

            if not missing_columns.empty:
                print(f"Table: {table_name} has missing values in columns: {missing_columns.index.tolist()}")
                for col in missing_columns.index:
                    mode_value = df[col].mode()[0]  # Get mode (most frequent value)
                    df[col].fillna(mode_value, inplace=True)
                    print(f"Filled missing values in column '{col}' with mode '{mode_value}'")
                
                self.dataframes_dict[table_name] = df

    def convert_string_to_datetime(self):
        """
        Convert any string columns that resemble date formats into datetime objects.
        """
        
        warnings.simplefilter(action='ignore', category=UserWarning)
        
        for table_name, df in self.dataframes_dict.items():
            object_columns = df.select_dtypes(include=['object']).columns
            for col in object_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"Converted '{col}' to datetime in table '{table_name}'")
                except (ValueError, TypeError):
                    print(f"Column '{col}' in table '{table_name}' is not a valid date format.")
            self.dataframes_dict[table_name] = df

    def process_feature_groups(self, fs, ver, dataframes=None, batch_size=5):
        """
        Create or retrieve feature groups and insert data in batches to avoid exceeding
        Hopsworks' limit of 5 parallel job executions.
        
        :param fs: Feature Store object to interact with feature groups.
        :param ver: Version of the feature group.
        :param dataframes: Dictionary of table names to DataFrames. Defaults to self.dataframes_dict.
        :param batch_size: The number of DataFrames to process at a time. Defaults to 5.
        """
        if dataframes is None:
            dataframes = self.dataframes_dict

        create_feature_groups(fs, ver, dataframes, batch_size)

    # def process_feature_groups(self, fs, ver, dataframes=None):
    #     """
    #     Create or retrieve feature groups and insert data only if it is a new feature group.
    #     :param fs: Feature Store object to interact with feature groups.
    #     """
    #     if dataframes is None:
    #         dataframes = self.dataframes_dict
        
    #     for table_name, df in dataframes.items():
    #         feature_group_name = f"{table_name}_fg"
    #         primary_key = ['id']
    #         event_time_column = 'event_time'
            
    #         try:
    #             # Attempt to retrieve the existing feature group
    #             feature_group = fs.get_feature_group(name=feature_group_name, version=ver)
    #             print(f"Feature group '{feature_group_name}' already exists. Skipping creation and insertion.")
    #         except Exception as e:
    #             # Handle any exception that occurs during retrieval (e.g., group not found)
    #             print(f"Feature group '{feature_group_name}' not found. Creating a new feature group. Error: {e}")
                
    #             # Create a new feature group
    #             feature_group = fs.create_feature_group(
    #                 name=feature_group_name,
    #                 version=ver,
    #                 description=f"Feature group for {feature_group_name}",
    #                 primary_key=primary_key,
    #                 event_time=event_time_column
    #             )
                
    #             # Insert the DataFrame into the newly created feature group
    #             feature_group.insert(df)
    #             print(f"Inserted data into new feature group: {feature_group_name}")
                
    def save_dataframes_to_db(self, if_exists='replace'):
        """
        Save each DataFrame in the dictionary as a table in the database.
        :param if_exists: Behavior when the table already exists. Options:
                        'fail', 'replace', 'append'. Default is 'replace'.
        """
        if self.engine is None:
            print("error connecting to database")
        for table_name, df in self.dataframes_dict.items():
            try:
                # Save each DataFrame as a table in the database
                df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=False)
                print(f"Table '{table_name}'_cleaned successfully saved to the database.")
            except Exception as e:
                print(f"Error saving table '{table_name}_cleaned': {e}")
    