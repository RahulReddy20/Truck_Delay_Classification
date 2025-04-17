from utils.database_utils import get_table_names, connect
from utils.config_utils import get_db_config, read_config
from utils.feature_group_utils import hopswork_login, fetch_df_from_feature_groups, create_feature_groups
from constants import CONFIG_FILE_PATH
from components.data_preparation import DataPreparation

STAGE_NAME = "Data Preparation"


class DataPreparationPipeline():
    def __init__(self):
        pass

    def main(self):
        engine = connect(get_db_config(read_config(CONFIG_FILE_PATH)))
        table_names = get_table_names(engine)

        project = hopswork_login()
        fs = project.get_feature_store()

        df_dict = fetch_df_from_feature_groups(fs, table_names, 2)
        data_preparation_obj = DataPreparation()

        df_dict = data_preparation_obj.drop_event_time_column(
            df_dict)  

        subsets_to_drop_duplicates = {
            'city_weather_fg': ['city_id', 'date', 'hour'],
            'routes_weather_fg': ['route_id', 'date'],
            'trucks_table_fg': ['truck_id'],
            'drivers_table_fg': ['driver_id'],
            'routes_table_fg': ['route_id', 'destination_id', 'origin_id'],
            'truck_schedule_table_fg': ['truck_id', 'route_id', 'departure_date'],
            'traffic_table_fg': ['route_id', 'date', 'hour']
        }
        df_dict = data_preparation_obj.drop_duplicates_in_dataframes(
            df_dict, subsets_to_drop_duplicates)  

        unnecessary_columns_dict = {
            'city_weather_fg': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder'],
            'routes_weather_fg': ['chanceofrain', 'chanceoffog', 'chanceofsnow', 'chanceofthunder']
        }
        df_dict = data_preparation_obj.drop_columns(
            df_dict, unnecessary_columns_dict)

        df_dict['city_weather_fg'] = data_preparation_obj.add_datetime_column(
            df_dict['city_weather_fg'], 'date', 'hour')
        df_dict['traffic_table_fg'] = data_preparation_obj.add_datetime_column(
            df_dict['traffic_table_fg'], 'date', 'hour')

        schedule_df = data_preparation_obj.adjust_datetime(
            df_dict['truck_schedule_table_fg'].copy(), 'departure_date', 'estimated_arrival')
        schedule_df = data_preparation_obj.create_date_range(
            schedule_df, 'departure_date', 'estimated_arrival', 'date')
        schedule_route_weather_df = data_preparation_obj.merge_data(
            schedule_df, df_dict['routes_weather_fg'], ['route_id', 'date'])
        schedule_route_weather_df = data_preparation_obj.aggregate_weather(
            schedule_route_weather_df, ['id_x', 'truck_id', 'route_id']).rename(columns={'id_x': 'id'})
        schedule_route_weather_df = data_preparation_obj.merge_data(
            df_dict['truck_schedule_table_fg'], schedule_route_weather_df, ['id', 'truck_id', 'route_id'])
        # print(schedule_route_weather_df.isna().sum())

        nearest_hour_schedule_df = data_preparation_obj.round_datetime_columns(
            df_dict['truck_schedule_table_fg'], ['estimated_arrival', 'departure_date'], freq='H')
        nearest_hour_schedule_route_df = data_preparation_obj.merge_data(
            nearest_hour_schedule_df, df_dict['routes_table_fg'], on_cols='route_id', how='left')

        origin_weather_data_df = df_dict['city_weather_fg'].copy()
        destination_weather_data_df = df_dict['city_weather_fg'].copy()

        nearest_hour_schedule_route_origin_weather_df = data_preparation_obj.merge_data(
            nearest_hour_schedule_route_df,
            origin_weather_data_df,
            left_on=['departure_date_nearest_hour', 'origin_id'],
            right_on=['date_time', 'city_id'],
            how='left'
        )
        nearest_hour_schedule_route_origin_weather_df = data_preparation_obj.drop_columns(
            nearest_hour_schedule_route_origin_weather_df,
            ['id']
        )
        origin_destination_weather_merge_df = data_preparation_obj.merge_data(
            nearest_hour_schedule_route_origin_weather_df,
            destination_weather_data_df,
            left_on=['estimated_arrival_nearest_hour', 'destination_id'],
            right_on=['date_time', 'city_id'],
            how='left'
        )
        # print(origin_destination_weather_merge_df)
        # print(origin_destination_weather_merge_df.isna().sum())

        truck_schedule_for_traffic_df = data_preparation_obj.round_datetime_columns(
            df_dict['truck_schedule_table_fg'],
            ['departure_date', 'estimated_arrival'],
            freq='H'
        )
        hourly_exploded_scheduled_df = data_preparation_obj.explode_date_ranges(
            truck_schedule_for_traffic_df,
            'departure_date', 'estimated_arrival',
            'custom_date', freq='H'
        )
        scheduled_traffic_df = data_preparation_obj.merge_data(
            hourly_exploded_scheduled_df,
            df_dict['traffic_table_fg'],
            left_on=['route_id', 'custom_date'],
            right_on=['route_id', 'date_time']
        )
        scheduled_route_traffic_merge_df = data_preparation_obj.aggregate_route_traffic_data(
            scheduled_traffic_df,
            group_cols=['id_x', 'truck_id', 'route_id']
        )

        origin_destination_weather_schedule_traffic_merge = data_preparation_obj.merge_data(
            origin_destination_weather_merge_df,
            scheduled_route_traffic_merge_df,
            on_cols=['id_x', 'truck_id', 'route_id'],
            how='left'
        )
        origin_destination_weather_schedule_traffic_merge = data_preparation_obj.drop_columns(
            origin_destination_weather_schedule_traffic_merge,
            ['id_y']
        )
        merged_data_weather_traffic = data_preparation_obj.merge_data(
            schedule_route_weather_df,
            origin_destination_weather_schedule_traffic_merge,
            left_on=['id', 'truck_id', 'route_id',
                     'departure_date', 'estimated_arrival', 'delay'],
            right_on=['id_x', 'truck_id', 'route_id',
                      'departure_date', 'estimated_arrival', 'delay'],
            how='left'
        )
        merged_data_weather_traffic_trucks = data_preparation_obj.merge_data(
            merged_data_weather_traffic,
            df_dict['trucks_table_fg'],
            on_cols='truck_id',
            how='left'
        )

        merged_data_weather_traffic_trucks = data_preparation_obj.drop_columns(
            merged_data_weather_traffic_trucks, ['id'])

        final_merge = data_preparation_obj.merge_data(
            merged_data_weather_traffic_trucks,
            df_dict['drivers_table_fg'],
            left_on='truck_id',
            right_on='vehicle_no',
            how='left'
        )
        final_merge = data_preparation_obj.apply_has_midnight(
            final_merge,
            'is_midnight',
            'departure_date',
            'estimated_arrival'
        )

        # print(final_merge)
        # print(final_merge.isna().sum())

        final_merge = final_merge.loc[:, ~final_merge.columns.duplicated()]

        rename_column_names = {
            'id_x': 'unique_id',
            'temp_x': 'origin_temp',
            'wind_speed_x': 'origin_wind_speed',
            'description_x': 'origin_description',
            'precip_x': 'origin_precip',
            'humidity_x': 'origin_humidity',
            'visibility_x': 'origin_visibility',
            'pressure_x': 'origin_pressure',
            'temp_y': 'destination_temp',
            'wind_speed_y': 'destination_wind_speed',
            'description_y': 'destination_description',
            'precip_y': 'destination_precip',
            'humidity_y': 'destination_humidity',
            'visibility_y': 'destination_visibility',
            'pressure_y': 'destination_pressure',
        }
        drop_columns = ['date_time_x', 'city_id_x',
                        'id_y', 'date_time_y', 'city_id_y', 'id']

        final_merge = data_preparation_obj.drop_columns(
            final_merge, drop_columns)
        final_merge = data_preparation_obj.rename_columns(
            final_merge, rename_column_names)
        print(final_merge)
        print(final_merge.info())
        
        final_merge = final_merge.dropna()

        data_preparation_obj.add_event_time_column(final_merge)
        create_feature_groups(fs, 1, {'final_merge_fg': final_merge})


if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :", STAGE_NAME)
        obj = DataPreparationPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e
