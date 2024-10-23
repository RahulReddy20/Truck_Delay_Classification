import numpy as np
import pandas as pd
from utils.feature_group_utils import fetch_df_from_feature_groups
from datetime import datetime


class DataPreparation():
    def __init__(self):
        pass

    def drop_event_time_column(self, feature_dataframes):
        for key, df in feature_dataframes.items():
            if 'event_time' in df.columns:
                df.drop(columns=['event_time'], inplace=True)
        return feature_dataframes

    def drop_duplicates_in_dataframes(self, feature_dataframes, subsets):
        for df_name, subset in subsets.items():
            feature_dataframes[df_name] = feature_dataframes[df_name].drop_duplicates(
                subset=subset, inplace=False)

        return feature_dataframes

    def drop_columns(self, dataframes, columns, inplace=False):
        if isinstance(dataframes, dict):
            if isinstance(columns, dict):
                for df_key, columns_to_drop in columns.items():
                    if df_key in dataframes:
                        dataframes[df_key] = dataframes[df_key].drop(
                            columns=columns_to_drop, inplace=inplace)
            else:
                raise ValueError(
                    "When dataframes is a dictionary, columns should be a dictionary too.")
        else:
            if isinstance(columns, list):
                dataframes = dataframes.drop(columns=columns, inplace=inplace)
            else:
                raise ValueError(
                    "When dataframes is a single dataframe, columns should be a list.")

        return dataframes

    def add_datetime_column(self, dataframe, date_column, hour_column):
        # Create a new 'date_time' column by adding 'date' and 'hour'
        dataframe['date_time'] = dataframe[date_column] + \
            pd.to_timedelta(dataframe[hour_column] // 100, unit='h')

        # Drop the original 'date' and 'hour' columns
        dataframe = dataframe.drop(columns=[date_column, hour_column])

        # Insert the 'date_time' column at the second position
        dataframe.insert(1, 'date_time', dataframe.pop('date_time'))

        return dataframe

    def adjust_datetime(self, df, dep_col, arr_col):
        df[dep_col] = df[dep_col].dt.floor('6H')
        df[arr_col] = df[arr_col].dt.ceil('6H')
        return df

    def create_date_range(self, df, dep_col, arr_col, new_col):
        df[new_col] = df.apply(lambda row: pd.date_range(
            start=row[dep_col], end=row[arr_col], freq='6H'), axis=1)
        return df.explode(new_col)

    def merge_data(self, df_left, df_right, on_cols=None, left_on=None, right_on=None, how='left'):
        if on_cols:
            # Use the same columns for both dataframes to join on
            return pd.merge(df_left, df_right, on=on_cols, how=how)
        elif left_on and right_on:
            # Use different columns to join from left and right dataframes
            return pd.merge(df_left, df_right, left_on=left_on, right_on=right_on, how=how)
        else:
            raise ValueError(
                "Either 'on_cols' or both 'left_on' and 'right_on' must be provided.")

    def aggregate_weather(self, df, group_cols):
        return df.groupby(group_cols, as_index=False).agg(
            route_avg_temp=('temp', 'mean'),
            route_avg_wind_speed=('wind_speed', 'mean'),
            route_avg_precip=('precip', 'mean'),
            route_avg_humidity=('humidity', 'mean'),
            route_avg_visibility=('visibility', 'mean'),
            route_avg_pressure=('pressure', 'mean'),
            route_description=('description', lambda x: x.mode(
            ).iloc[0] if not x.mode().empty else np.nan)
        )

    def round_datetime_columns(self, df, datetime_cols, freq='H'):
        df_copy = df.copy()
        for col in datetime_cols:
            df_copy[f"{col}_nearest_hour"] = df_copy[col].dt.round(freq)
        return df_copy

    def custom_accident_agg(self, values):
        return 1 if any(values == 1) else 0

    def aggregate_route_traffic_data(self, df, group_cols):
        return df.groupby(group_cols, as_index=False).agg(
            avg_no_of_vehicles=('no_of_vehicles', 'mean'),
            accident=('accident', self.custom_accident_agg)
        )

    def explode_date_ranges(self, df, start_col, end_col, new_col, freq='H'):
        df[new_col] = [pd.date_range(start, end, freq=freq)
                       for start, end in zip(df[start_col], df[end_col])]
        return df.explode(new_col, ignore_index=True)

    def apply_has_midnight(self, df, result_col, start_col, end_col):
        df[result_col] = df.apply(lambda row: self.has_midnight(
            row[start_col], row[end_col]), axis=1)
        return df

    def has_midnight(self, start, end):
        return int(start.date() != end.date())

    def rename_columns(self, df, columns_dict):
        return df.rename(columns=columns_dict)

    def add_event_time_column(self, final_merge):
        today_date = datetime.now().date()
        if 'event_time' not in final_merge.columns:
            final_merge['event_time'] = [today_date] * \
                len(final_merge)  # Add 'event_time' column
            print(f"'event_time' column added for table")
        else:
            print(f"'event_time' column already exists in table")
