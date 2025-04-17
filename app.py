import streamlit as st
import pandas as pd
import joblib
import hopsworks
import os
import mlflow
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.feature_group_utils import hopswork_login, fetch_df_from_feature_groups


project = hopswork_login()
fs = project.get_feature_store() # Login to Hopsworks
mr = project.get_model_registry()  # Model Registry

table_names = ['final_merge']

df_dict = fetch_df_from_feature_groups(fs, table_names, 1) 
final_merge = df_dict['final_merge_fg']  


def load_model_and_encoders():
    xgb_model = mr.get_model("XGBoost", version=4)  
    model_dir = xgb_model.download()
    trained_model = joblib.load(os.path.join(model_dir, "XGBoost_model.pkl"))

    scaler_model = mr.get_model("standardScaler", version=1)  
    scaler_dir = scaler_model.download()
    standardScaler = joblib.load(os.path.join(scaler_dir, "standardScaler.pkl"))

    ratings_encoder_model = mr.get_model("ratings_label_encoder", version=1) 
    ratings_encoder_dir = ratings_encoder_model.download()
    ratings_label_encoder = joblib.load(os.path.join(ratings_encoder_dir, "ratings_label_encoder.pkl"))

    onehot_encoder_model = mr.get_model("onehotEncoder", version=1)  
    onehot_encoder_dir = onehot_encoder_model.download()
    onehotEncoder = joblib.load(os.path.join(onehot_encoder_dir, "onehotEncoder.pkl"))

    is_midnight_encoder_model = mr.get_model("is_midnight_label_encoder", version=1)  
    is_midnight_encoder_dir = is_midnight_encoder_model.download()
    is_midnight_label_encoder = joblib.load(os.path.join(is_midnight_encoder_dir, "is_midnight_label_encoder.pkl"))

    accident_encoder_model = mr.get_model("accident_label_encoder", version=1)  
    accident_encoder_dir = accident_encoder_model.download()
    accident_label_encoder = joblib.load(os.path.join(accident_encoder_dir, "accident_label_encoder.pkl"))

    return trained_model, standardScaler, ratings_label_encoder, onehotEncoder, is_midnight_label_encoder, accident_label_encoder

st.title('Truck Delay Classification')

options = ['date_filter', 'truck_id_filter', 'route_id_filter']
selected_option = st.radio("Choose an option:", options)

if selected_option == 'date_filter':
    st.write("### Date Ranges")
    from_date = st.date_input("Enter start date:", value=min(final_merge['departure_date']))
    to_date = st.date_input("Enter end date:", value=max(final_merge['departure_date']))

elif selected_option == 'truck_id_filter':
    st.write("### Truck ID")
    truck_id = st.selectbox('Select truck ID:', final_merge['truck_id'].unique())

elif selected_option == 'route_id_filter':
    st.write("### Route ID")
    route_id = st.selectbox('Select route ID:', final_merge['route_id'].unique())

if st.button('Predict'):
    try:
        filter_query = None
        if selected_option == 'date_filter':
            filter_query = (final_merge['departure_date'] >= str(from_date)) & (final_merge['departure_date'] <= str(to_date))
            filtered_data = final_merge[filter_query]
            sentence = f"Truck delay predictions during the date range {from_date} to {to_date}."

        elif selected_option == 'truck_id_filter':
            filter_query = (final_merge['truck_id'] == truck_id)
            filtered_data = final_merge[filter_query]
            sentence = f"Truck delay predictions for Truck ID {truck_id}."

        elif selected_option == 'route_id_filter':
            filter_query = (final_merge['route_id'] == str(route_id))
            filtered_data = final_merge[filter_query]
            sentence = f"Truck delay predictions for Route ID {route_id}."

        else:
            st.write("Please select at least one filter.")
            filtered_data = pd.DataFrame()  

        if not filtered_data.empty:
            st.write(sentence)
            output_columns = filtered_data[['truck_id', 'route_id', 'departure_date']].reset_index(drop=True)
            trained_model, standardScaler, ratings_label_encoder, onehotEncoder, is_midnight_label_encoder, accident_label_encoder = load_model_and_encoders()
            
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
            filtered_data= filtered_data[feature_cols]

            categorical_columns = ['route_description', 'origin_description', 'destination_description', 'fuel_type', 'gender', 'driving_style']
            label_encoder_columns = ['accident', 'ratings','is_midnight']

            encoded_cats = onehotEncoder.transform(filtered_data[categorical_columns])
            encoded_features = list(onehotEncoder.get_feature_names_out(categorical_columns))
            filtered_data[encoded_features] = encoded_cats
            # encoded_cats_df = pd.DataFrame(encoded_cats, columns=onehotEncoder.get_feature_names_out(categorical_columns))
            filtered_data =  filtered_data.drop(columns=categorical_columns, axis=1)

            filtered_data['accident'] = accident_label_encoder.transform(filtered_data['accident'])
            filtered_data['ratings'] = ratings_label_encoder.transform(filtered_data['ratings'])
            filtered_data['is_midnight'] = is_midnight_label_encoder.transform(filtered_data['is_midnight'])

            filtered_data[cts_cols] = standardScaler.transform(filtered_data[cts_cols])
            # scaled_cont_df = pd.DataFrame(scaled_cont, columns=cts_cols)

            X = filtered_data
            
            predictions = trained_model.predict(X)
            output_columns['delay_prediction'] = predictions

            st.write(output_columns)

    except Exception as e:
        st.error(f"An error occurred: {e}")
